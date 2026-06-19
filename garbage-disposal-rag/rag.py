"""
大田区 ルール解説RAG — MVP スキャフォールド
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

実行 (あなたの環境で / uv プロジェクト):
  uv sync                                 # 依存を .venv に導入
  uv run python rag.py ingest    # HTML取得→チャンク→埋め込み→ota_index.npz
  uv run python rag.py eval      # 内蔵ゴールドセットで groundedness/棄却 を確認
  uv run python rag.py ask "粗大ごみの申込方法は?"

Cloudflare移植:
  埋め込み -> Workers AI (@cf/baai/bge-m3) / 索引 -> Vectorize / メタdata -> D1 / 推論 -> Workers AI or Anthropic API。
  まずローカルで eval ループを回して retrieval を詰めてから移植するのが速い。
"""

import sys, re, json, time, os
from dataclasses import dataclass, asdict

import anthropic
import numpy as np
import requests
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer

# ---- コーパス: 大田区サイトのHTMLページ (ルール解説のみ) -------------------
# category はルーティングと出典表示に使う。収集日ページは「タイミング規則」部分のみ採用。
SOURCES = [
    ("プラスチック",        "https://www.city.ota.tokyo.jp/seikatsu/gomi/shigentogomi/index.html"),
    ("可燃ごみ・不燃ごみ",  "https://www.city.ota.tokyo.jp/seikatsu/gomi/shigentogomi/gomishigen.html"),
    ("粗大ごみ",            "https://www.city.ota.tokyo.jp/seikatsu/gomi/shigentogomi/index.html"),
    ("リチウムイオン電池",  "https://www.city.ota.tokyo.jp/seikatsu/gomi/shigentogomi/index.html"),
    ("区では収集できないもの","https://www.city.ota.tokyo.jp/seikatsu/gomi/shigentogomi/index.html"),
    ("出し方の基本ルール",  "https://www.city.ota.tokyo.jp/seikatsu/gomi/shigentogomi/gomishigen.html"),
    ("事業系ごみ",          "https://www.city.ota.tokyo.jp/seikatsu/gomi/gomi/kushuu.html"),
]
# 注: 上の index.html は目次ページ。実運用では各小ページ (プラスチック/粗大ごみ等の個別URL) に
#     差し替えると粒度が上がる。MVPはまず動かす。PDF (令和8年度版) は穴埋め用に後から追加。

PDF_SUPPLEMENT = "https://www.city.ota.tokyo.jp/seikatsu/gomi/shigentogomi/katei-shigen-gomi_pamphlet.files/08shigenntogominowakekatadashikata.pdf"

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
            html = requests.get(url, timeout=20).text
        except Exception as e:
            print(f"[skip] {url}: {e}"); continue
        soup = BeautifulSoup(html, "html.parser")
        main = soup.find(id="main") or soup.find("article") or soup.body
        if not main:
            continue
        # 見出し(h2/h3)でセクション分割し、見出しを各チャンク先頭に残す
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
        for el in main.find_all(["h2", "h3", "p", "li", "td"]):
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

def embed(texts, model, kind):
    # e5系は "query: " / "passage: " のprefixが必須。付け忘れると検索品質が静かに落ちる。
    prefix = "query: " if kind == "query" else "passage: "
    return model.encode([prefix + t for t in texts], normalize_embeddings=True)

def ingest():
    chunks = fetch_and_chunk()
    model = SentenceTransformer(EMB_MODEL)
    vecs = embed([c.text for c in chunks], model, "passage")
    meta = [asdict(c) for c in chunks]
    np.savez(INDEX_PATH, vecs=np.asarray(vecs, dtype="float32"),
             meta=np.array(json.dumps(meta, ensure_ascii=False)))
    print(f"[ingest] saved -> {INDEX_PATH}")

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
