"""
大田区 ゴミ出しルール解説RAG
================================================

スコープ:
  - ○ ルール解説 (粗大ごみの申込方法/可燃ごみは何時まで/リチウムイオン電池の出し方 など)
  - × 分別判定 「これは何ゴミ?」 -> 品目→区分のルックアップ問題。別系統。入口で弾く。
  - × 収集日 「うちの収集日は?」 -> カレンダー+集積所看板依存。入口で弾く。

設計判断 (大田区コーパス固有):
  1. HTML優先。メインPDFは縦書き多段組で抽出が崩れるため補助に回す。
  2. セクション単位チャンク + 見出しをチャンク先頭に残す (区分の意味喪失を防ぐ)。
  3. 棄却 (abstention) を一級市民として実装。確信度が低ければ答えない。
  4. 出典URL + last_verified を必ず回答に添える (改定が頻繁なドメイン)。
"""

import sys, re, json, time, os, random
from dataclasses import dataclass, asdict

import anthropic
import numpy as np
import requests
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer

# ---- コーパス: 大田区「家庭から出る資源とごみ」(index.html) 配下の全ページ -----
# category はルーティングと出典表示に使う（= リンクテキスト）。
BASE = "https://www.city.ota.tokyo.jp"
UA = "Mozilla/5.0 (compatible; ota-gomi-rag/0.1; personal research)"  # gov サイトの既定UA弾き対策

_PAGES = [
    ("環境ポスターを載せて清掃車が走行中！", "/seikatsu/gomi/shigentogomi/sisoussyaposuta_.html"),
    ("資源とごみの集積所について", "/seikatsu/gomi/shigentogomi/manners.html"),
    ("豪雨が発生した際の家庭ごみ・事業系ごみの取扱いについて", "/seikatsu/gomi/shigentogomi/gouuhasseijinogomisyuusyuunitsui.html"),
    ("令和7年9月11日大田区豪雨により発生したごみの取扱いについて", "/seikatsu/gomi/shigentogomi/gouusaigaijinotaiou.html"),
    ("パンフレット「令和8年度版 資源とごみの分け方・出し方」", "/seikatsu/gomi/shigentogomi/katei-shigen-gomi_pamphlet.html"),
    ("パンフレット・啓発冊子等", "/seikatsu/gomi/shigentogomi/panfu.html"),
    ("資源とごみの収集日", "/seikatsu/gomi/shigentogomi/gomishigen.html"),
    ("資源（7品目）", "/seikatsu/gomi/shigentogomi/shigen.html"),
    ("プラスチック", "/seikatsu/gomi/shigentogomi/purasuhcikunodashikata.html"),
    ("可燃ごみ", "/seikatsu/gomi/shigentogomi/kanen.html"),
    ("不燃ごみ", "/seikatsu/gomi/shigentogomi/funen.html"),
    ("粗大ごみ", "/seikatsu/gomi/shigentogomi/sodai.html"),
    ("家電（エアコン、テレビ、洗濯機・衣類乾燥機、冷蔵・冷凍庫）のリサイクル", "/seikatsu/gomi/shigentogomi/kaden.html"),
    ("小型家電（携帯電話、デジカメ等）リサイクル事業について", "/seikatsu/gomi/shigentogomi/kogatakadenn.html"),
    ("小型充電式電池 (リチウムイオン電池等) の処分方法について", "/seikatsu/gomi/shigentogomi/kogata.html"),
    ("家庭用パーソナルコンピューターのリサイクル", "/seikatsu/gomi/shigentogomi/katei.html"),
    ("古着の拠点回収", "/seikatsu/gomi/shigentogomi/konuno.html"),
    ("常設ボックスによる古着の回収", "/seikatsu/gomi/shigentogomi/furugikaisyuubox.html"),
    ("不要品のリユース（再利用）", "/seikatsu/gomi/shigentogomi/oikura.html"),
    ("資源とごみの散乱防止について", "/seikatsu/gomi/shigentogomi/karasunetto.html"),
    ("家庭用使用済みインクカートリッジの回収について", "/seikatsu/gomi/shigentogomi/ink.html"),
    ("廃食用油の出し方について", "/seikatsu/gomi/shigentogomi/haishoku.html"),
    ("大田区有料ごみ処理券取り扱い店舗", "/seikatsu/gomi/shigentogomi/gomi-syori-ken_shop-list.html"),
    ("臨時ごみ（一度に多量のごみを出す場合）について", "/seikatsu/gomi/shigentogomi/rinji.html"),
    ("引越しを予定されている皆様へ...ごみの出し方のお知らせ", "/seikatsu/gomi/shigentogomi/hikkoshi-no-gomi.html"),
    ("区では収集できないもの", "/seikatsu/gomi/shigentogomi/syuusyuu.html"),
    ("ペット・動物死体の引き取り", "/seikatsu/gomi/shigentogomi/doubutsushitai.html"),
    ("強風時の資源とごみの出し方", "/seikatsu/gomi/shigentogomi/kyoufuji.html"),
    ("使い捨てライターの出し方について", "/seikatsu/gomi/shigentogomi/disposable-cigarette-lighter.html"),
    ("スプレー缶、カセットボンベの出し方について", "/seikatsu/gomi/shigentogomi/supureikan_kasai.html"),
    ("在宅医療廃棄物について", "/seikatsu/gomi/shigentogomi/zaitaku-iryou-haikibutsu.html"),
    ("消火器のリサイクル", "/seikatsu/gomi/shigentogomi/shoukaki_recycle.html"),
    ("自動車とオートバイのリサイクル", "/seikatsu/gomi/shigentogomi/jidousha.html"),
    ("水銀を含むごみの出し方について", "/seikatsu/gomi/shigentogomi/suigin.html"),
    ("基準を超える石綿（アスベスト）を含む珪藻土製品のメーカー回収について", "/seikatsu/gomi/shigentogomi/keisoudo.html"),
    ("清掃だより(令和8年6月号)を発行しました", "/seikatsu/gomi/shigentogomi/seisoudayori.html"),
    ("ごみの戸別訪問収集", "/seikatsu/gomi/shigentogomi/houmonshushu.html"),
    ("ごみ・資源の持ち去り防止対策", "/seikatsu/gomi/keikaku_jisseki/mochisariboushi.html"),
    ("不用品回収業者にご注意ください", "/seikatsu/gomi/shigentogomi/kaisyuu-gyousya.html"),
    ("雑がみ回収袋の作り方", "/seikatsu/gomi/shigentogomi/zatsugamikaisyubukuronotsukurika.html"),
    ("コードレス掃除機用非純正のバッテリーパックについて", "/seikatsu/gomi/shigentogomi/codeles-cleaner_battery-pack.html"),
]
SOURCES = [(category, BASE + path) for category, path in _PAGES]

EMB_MODEL = "intfloat/multilingual-e5-base"  # JA特化なら cl-nagoya/ruri-base に差替 (prefix方式が異なる点に注意)
INDEX_PATH = "ota_index.npz"
MAX_CHARS = 700          # セクションが長い場合の分割上限
SCORE_FLOOR = 0.78       # これ未満なら棄却 (e5正規化cos。要校正)
TOP_K = 4
LLM_MODEL = os.environ.get("OTA_LLM_MODEL", "claude-haiku-4-5")  # 生成モデル(安価なPoC既定)

# ---- 境界ルーター: スコープ外クエリを入口で弾く ----------------------------
ITEM_CLASSIFY_PAT = re.compile(r"(は何ごみ|は何ゴミ|何ごみ\?|何ゴミ\?|どのごみ|どう捨て|捨て方は\s*$)")
COLLECTION_DAY_PAT = re.compile(r"(収集日|何曜日|いつ出せ|いつ収集|回収日)")

def route(query: str) -> str:
    if COLLECTION_DAY_PAT.search(query):
        return "collection_day"   # カレンダー依存。RAG非対象。
    if ITEM_CLASSIFY_PAT.search(query):
        return "item_classify"    # ルックアップ問題。RAG非対象。
    return "rule_explanation"

OUT_OF_SCOPE_MSG = {
    "collection_day": "収集日は集積所の看板または大田区ごみ分別アプリでご確認ください（住所により異なります）。",
    "item_classify":  "品目ごとの分別区分は大田区ごみ分別アプリの分別辞典でご確認ください。本回答は出し方ルールの解説に限定しています。",
}

# ---- 取り込み --------------------------------------------------------------
@dataclass
class Chunk:
    text: str
    category: str
    source_url: str
    section: str
    last_verified: str

def fetch_and_chunk():
    today = time.strftime("%Y-%m-%d")
    chunks = []
    for category, url in SOURCES:
        try:
            r = requests.get(url, timeout=20, headers={"User-Agent": UA})
            r.encoding = r.apparent_encoding or r.encoding
        except Exception as e:
            print(f"[skip] {url}: {e}"); continue
        soup = BeautifulSoup(r.text, "html.parser")
        main = soup.find("main")   # 大田区は <main> タグ内に本文を配置
        if main is None or not main.get_text(strip=True):
            print(f"[skip] {url}: <main> が無い/空")
            continue
        # 見出し(h2/h3)でセクション分割し、見出しを各チャンク先頭に残す。
        # <table> は別扱いで「行を分断しない」チャンクにする（td のフラット走査では行が崩れるため）。
        section = category
        buf = []
        def flush():
            nonlocal buf
            body = "\n".join(b for b in buf if b).strip()
            if len(body) < 30:
                buf = []; return
            for piece in split_keep_header(section, body):
                chunks.append(Chunk(piece, category, url, section, today))
            buf = []
        for el in main.find_all(["h2", "h3", "p", "li", "table"]):
            if el.name != "table" and el.find_parent("table"):
                continue   # 表の中身は table 側でまとめるのでスキップ
            if el.name == "table":
                flush()
                rows = _table_to_rows(el)
                for piece in split_table_keep_header(section, rows):
                    chunks.append(Chunk(piece, category, url, section, today))
                continue
            t = el.get_text(" ", strip=True)
            if not t:
                continue
            if el.name in ("h2", "h3"):
                flush(); section = t
            else:
                buf.append(t)
        flush()
    print(f"[ingest] {len(chunks)} chunks from {len(SOURCES)} pages")
    return chunks

def split_keep_header(header, body):
    """セクションが長ければ分割。各断片の先頭に見出しを必ず付与。"""
    out, cur = [], ""
    for sent in re.split(r"(?<=。)", body):
        if len(cur) + len(sent) > MAX_CHARS and cur:
            out.append(f"【{header}】{cur.strip()}"); cur = ""
        cur += sent
    if cur.strip():
        out.append(f"【{header}】{cur.strip()}")
    return out

def _table_to_rows(table):
    """<table> を「1行=セルを ｜ で連結した文字列」のリストにする（行は分断しない単位）。"""
    rows = []
    for tr in table.find_all("tr"):
        cells = [c.get_text(" ", strip=True) for c in tr.find_all(["th", "td"])]
        cells = [c for c in cells if c]
        if cells:
            rows.append(" ｜ ".join(cells))
    return rows

def split_table_keep_header(header, rows):
    """表を見出し付きチャンクにまとめる。MAX_CHARS 超過時は行境界で分割し、行は決して割らない。"""
    out, cur = [], []
    for row in rows:
        if cur and sum(len(r) for r in cur) + len(row) > MAX_CHARS:
            out.append(f"【{header}】\n" + "\n".join(cur)); cur = []
        cur.append(row)
    if cur:
        out.append(f"【{header}】\n" + "\n".join(cur))
    return out

def embed(texts, model, kind):
    # e5系は "query: " / "passage: " のprefixが必須。付け忘れると検索品質が静かに落ちる。
    prefix = "query: " if kind == "query" else "passage: "
    return model.encode([prefix + t for t in texts], normalize_embeddings=True)

def ingest():
    chunks = fetch_and_chunk()
    _dump_for_inspection(chunks)   # 観察用出力（崩れを後から目視するため）
    model = SentenceTransformer(EMB_MODEL)
    vecs = embed([c.text for c in chunks], model, "passage")
    meta = [asdict(c) for c in chunks]
    np.savez(INDEX_PATH, vecs=np.asarray(vecs, dtype="float32"),
             meta=np.array(json.dumps(meta, ensure_ascii=False)))
    print(f"[ingest] saved -> {INDEX_PATH}")

def _dump_for_inspection(chunks):
    """観察用の出力: コーパス取得日 / 全チャンクのダンプ / 無作為20件サンプル。"""
    today = time.strftime("%Y-%m-%d")
    with open("corpus_date.txt", "w", encoding="utf-8") as f:
        f.write(f"corpus snapshot date: {today}\n")
        f.write(f"sources: {len(SOURCES)} pages\n")
        f.write(f"chunks: {len(chunks)}\n")
    with open("chunks_dump.txt", "w", encoding="utf-8") as f:
        for i, c in enumerate(chunks):
            f.write(f"=== [{i}] category={c.category} | section={c.section} | {c.source_url}\n")
            f.write(c.text + "\n\n")
    rng = random.Random(42)   # seed固定で再現可能
    sample = rng.sample(chunks, min(20, len(chunks)))
    with open("chunks_sample.txt", "w", encoding="utf-8") as f:
        f.write(f"# 無作為20件サンプル（seed=42 / 全{len(chunks)}件中）\n\n")
        for c in sample:
            f.write(f"--- category={c.category} | section={c.section}\n{c.text}\n\n")
    print("[ingest] inspection dump -> corpus_date.txt / chunks_dump.txt / chunks_sample.txt")

# ---- 検索 + 棄却 + 回答 ----------------------------------------------------
def load_index():
    d = np.load(INDEX_PATH, allow_pickle=True)
    return d["vecs"], json.loads(str(d["meta"]))

def retrieve(query, model, vecs, meta):
    q = embed([query], model, "query")[0]
    scores = vecs @ q
    idx = np.argsort(-scores)[:TOP_K]
    return [(float(scores[i]), meta[i]) for i in idx]

def build_prompt(query, hits):
    ctx = "\n\n".join(
        f"[出典: {h['category']} / {h['source_url']} / 確認日{h['last_verified']}]\n{h['text']}"
        for _, h in hits
    )
    return (
        "あなたは大田区の資源とごみの出し方ルールの案内係です。"
        "以下の【参照情報】だけを根拠に答えてください。参照情報に書かれていないことは"
        "推測せず「区の案内では確認できませんでした」と述べ、ごみ減量推進課(03-5744-1628)を案内してください。"
        "回答末尾に必ず出典URLを示してください。\n\n"
        f"【参照情報】\n{ctx}\n\n【質問】{query}\n【回答】"
    )

def call_llm(prompt: str) -> str:
    """Anthropic Messages API でテキスト応答を返す。

    APIキーはコードに書かず os.environ["ANTHROPIC_API_KEY"] から読む（SDKが自動取得）。
    モデルは LLM_MODEL（環境変数 OTA_LLM_MODEL で上書き可）。
    """
    client = anthropic.Anthropic()   # ANTHROPIC_API_KEY を環境変数から取得
    resp = client.messages.create(
        model=LLM_MODEL,
        max_tokens=600,
        messages=[{"role": "user", "content": prompt}],
    )
    return "".join(b.text for b in resp.content if b.type == "text")


def ask(query, verbose=True):
    r = route(query)
    if r != "rule_explanation":
        return {"abstain": True, "reason": r, "answer": OUT_OF_SCOPE_MSG[r], "hits": []}
    model = SentenceTransformer(EMB_MODEL)
    vecs, meta = load_index()
    hits = retrieve(query, model, vecs, meta)
    top = hits[0][0] if hits else 0.0
    if top < SCORE_FLOOR:
        return {"abstain": True, "reason": "low_score", "top_score": top,
                "answer": "区の案内では確認できませんでした。ごみ減量推進課(03-5744-1628)へご確認ください。",
                "hits": hits}
    answer = call_llm(build_prompt(query, hits))   # スタブ。未実装なら eval は retrieval のみ評価
    return {"abstain": False, "top_score": top, "answer": answer, "hits": hits}

# ---- 評価: 内蔵ゴールドセット (retrieval + 棄却の正しさを測る) --------------
GOLD = [
    # (質問, 期待カテゴリ or None, 棄却すべきか)
    ("粗大ごみの申込方法は?",                 "粗大ごみ",            False),
    ("可燃ごみは何時までに出せばいい?",        "出し方の基本ルール",  False),
    ("リチウムイオン電池はどう捨てる?",        "リチウムイオン電池",  False),
    ("プラスチックの分別はいつから始まった?",  "プラスチック",        False),
    ("夜にごみを出してもいい?",               "出し方の基本ルール",  False),
    ("エアコンは区で収集してくれる?",          "区では収集できないもの", False),
    ("ペットボトルのキャップは何ゴミ?",        None,                  True),   # 分別判定→棄却
    ("東蒲田の収集日は何曜日?",               None,                  True),   # 収集日→棄却
]

def evaluate():
    model = SentenceTransformer(EMB_MODEL)
    vecs, meta = load_index()
    ok = 0
    for q, exp_cat, should_abstain in GOLD:
        r = route(q)
        if should_abstain:
            passed = (r != "rule_explanation")
            print(f"{'PASS' if passed else 'FAIL'} [棄却] {q} -> route={r}")
            ok += passed; continue
        hits = retrieve(q, model, vecs, meta)
        top_score, top_cat = hits[0][0], hits[0][1]["category"]
        passed = (top_score >= SCORE_FLOOR) and (exp_cat is None or top_cat == exp_cat)
        print(f"{'PASS' if passed else 'FAIL'} [{top_score:.3f}] {q} -> {top_cat} (期待:{exp_cat})")
        ok += passed
    print(f"\n{ok}/{len(GOLD)} passed")

# ---- CLI -------------------------------------------------------------------
if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "help"
    if cmd == "ingest":
        ingest()
    elif cmd == "eval":
        evaluate()
    elif cmd == "ask":
        res = ask(sys.argv[2])
        print(json.dumps(res, ensure_ascii=False, indent=2))
    else:
        print(__doc__)
