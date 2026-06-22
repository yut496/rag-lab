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

# ---- コーパス: 7カテゴリに対応する固定ページのみ（Phase 2 Hotfix で見出し流用を廃止） -----
# category はページ見出し（リンクテキスト）の流用をやめ、source_url -> 7カテゴリ の明示マッピングで決める。
BASE = "https://www.city.ota.tokyo.jp"
UA = "Mozilla/5.0 (compatible; ota-gomi-rag/0.1; personal research)"  # gov サイトの既定UA弾き対策

# 許可カテゴリ（委託者確定・増減禁止）。全チャンクの category はこの7値のいずれかに正規化される。
CATEGORIES = (
    "プラスチック",
    "可燃ごみ",
    "不燃ごみ",
    "粗大ごみ",
    "資源",
    "区で収集できないもの",
    "集積所",
)

# source_url -> 7カテゴリ の明示マッピング。category はこの辞書だけで決定する（見出しは使わない）。
URL_CATEGORY = {
    BASE + "/seikatsu/gomi/shigentogomi/purasuhcikunodashikata.html": "プラスチック",
    BASE + "/seikatsu/gomi/shigentogomi/kanen.html":                  "可燃ごみ",
    BASE + "/seikatsu/gomi/shigentogomi/funen.html":                  "不燃ごみ",
    BASE + "/seikatsu/gomi/shigentogomi/sodai.html":                  "粗大ごみ",
    BASE + "/seikatsu/gomi/shigentogomi/shigen.html":                 "資源",
    BASE + "/seikatsu/gomi/shigentogomi/syuusyuu.html":               "区で収集できないもの",
    BASE + "/seikatsu/gomi/shigentogomi/manners.html":                "集積所",
}
# 取得対象は7カテゴリ対応ページのみ（旧41ページ -> 7ページ。非該当ページは除外）。
SOURCES = sorted(URL_CATEGORY)

EMB_MODEL = "intfloat/multilingual-e5-base"  # JA特化なら cl-nagoya/ruri-base に差替 (prefix方式が異なる点に注意)
INDEX_PATH = "index.npz"
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
    for url in SOURCES:
        category = URL_CATEGORY.get(url)
        if category is None:
            print(f"[warn] {url}: 7カテゴリに未割当 -> スキップ（混入）")
            continue
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
    """観察用の出力: 取得日 / category集計 / 全チャンクのダンプ / 無作為20件サンプル(v2)。"""
    today = time.strftime("%Y-%m-%d")
    # category 集計（全チャンクが許可7カテゴリに収まっているかを確認・表示する）
    counts = {}
    for c in chunks:
        counts[c.category] = counts.get(c.category, 0) + 1
    table = [f"  {cat}: {counts.get(cat, 0)}" for cat in CATEGORIES]
    extra = {cat: n for cat, n in counts.items() if cat not in CATEGORIES}
    print("[ingest] category 集計（許可7カテゴリ）:")
    for line in table:
        print(line)
    if extra:
        print(f"[warn] 7カテゴリ外の category が混入: {extra}")
    else:
        print("[ingest] OK: 全チャンクが7カテゴリ内")

    with open("corpus_date.txt", "w", encoding="utf-8") as f:
        f.write(f"corpus snapshot date: {today}\n")
        f.write(f"sources: {len(SOURCES)} pages\n")
        f.write(f"chunks: {len(chunks)}\n")
        f.write("category counts (whitelist of 7):\n")
        for line in table:
            f.write(line + "\n")
        if extra:
            f.write(f"[warn] outside-whitelist categories: {extra}\n")
    with open("chunks_dump.txt", "w", encoding="utf-8") as f:
        for i, c in enumerate(chunks):
            f.write(f"=== [{i}] category={c.category} | section={c.section} | {c.source_url}\n")
            f.write(c.text + "\n\n")
    rng = random.Random(42)   # seed固定で再現可能
    sample = rng.sample(chunks, min(20, len(chunks)))
    with open("chunks_sample_v2.txt", "w", encoding="utf-8") as f:
        f.write(f"# 無作為20件サンプル v2（seed=42 / 全{len(chunks)}件中 / category正規化後）\n\n")
        for c in sample:
            f.write(f"--- category={c.category} | section={c.section}\n{c.text}\n\n")
    print("[ingest] inspection dump -> corpus_date.txt / chunks_dump.txt / chunks_sample_v2.txt")

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
