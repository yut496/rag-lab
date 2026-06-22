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

import sys
import json
import time
from dataclasses import dataclass, asdict

import anthropic
import numpy as np
import requests
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

from helper import (
    dump_for_inspection,
    write_observation_log,
    write_observation_summary,
    split_keep_header,
    table_to_rows,
    split_table_keep_header,
)

from constant import (
    UA,
    CATEGORIES,
    URL_CATEGORY,
    SOURCES,
    EMB_MODEL,
    INDEX_PATH,
    SCORE_FLOOR,
    TOP_K,
    LLM_MODEL,
    ITEM_CLASSIFY_PAT,
    COLLECTION_DAY_PAT,
    OUT_OF_SCOPE_MSG,
    TEXT_HEAD,
)


# ---- 境界ルーター: スコープ外クエリを入口で弾く ----------------------------
def route(query: str) -> str:
    if COLLECTION_DAY_PAT.search(query):
        return "collection_day"  # カレンダー依存。RAG非対象。
    if ITEM_CLASSIFY_PAT.search(query):
        return "item_classify"  # ルックアップ問題。RAG非対象。
    return "rule_explanation"


# ---- 取り込み --------------------------------------------------------------
@dataclass
class Chunk:
    text: str
    category: str
    source_url: str
    section: str
    last_verified: str


def ingest():
    # ingestion: 各ページを HTTP 取得し (category, url, html) を返す（ネットワークを使う）。
    pages = []
    for url in SOURCES:
        category = URL_CATEGORY.get(url)
        if category is None:
            print(f"[warn] {url}: 7カテゴリに未割当 -> スキップ（混入）")
            continue
        try:
            r = requests.get(url, timeout=20, headers={"User-Agent": UA})
            r.encoding = r.apparent_encoding or r.encoding
        except Exception as e:
            print(f"[skip] {url}: {e}")
            continue
        pages.append((category, url, r.text))  # 生HTMLのまま渡す（パースは chunk 側）
    print(f"[ingest] {len(pages)} pages fetched")
    return pages


def chunk(pages):
    # chunking: 取得済み (category, url, html) を <main>抽出→見出し分割でチャンク化する（ネットワーク不要）。
    today = time.strftime("%Y-%m-%d")
    chunks = []
    for category, url, html in pages:
        soup = BeautifulSoup(html, "html.parser")
        main = soup.find("main")  # 大田区は <main> タグ内に本文を配置
        if main is None or not main.get_text(strip=True):
            print(f"[skip] {url}: <main> が無い/空")
            continue
        # 見出し(h2/h3)でセクション分割し、見出しを各チャンク先頭に残す。
        # <table> は別扱いで「行を分断しない」チャンクにする（td のフラット走査では行が崩れるため）。
        section = category
        buf = []

        def flush():
            # bufに溜め込んだ本文をchunkに吐き出す。
            body = "\n".join(b for b in buf if b).strip()
            if len(body) < 30:
                buf.clear()
                return
            for piece in split_keep_header(section, body):
                chunks.append(Chunk(piece, category, url, section, today))
            buf.clear()

        for el in main.find_all(["h2", "h3", "p", "li", "table"]):
            if el.name != "table" and el.find_parent("table"):
                continue  # 表の中身は table 側でまとめるのでスキップ
            if el.name == "table":
                flush()
                rows = table_to_rows(el)
                for piece in split_table_keep_header(section, rows):
                    chunks.append(Chunk(piece, category, url, section, today))
                continue
            t = el.get_text(" ", strip=True)
            if not t:
                continue
            if el.name in ("h2", "h3"):
                flush()
                section = t
            else:
                buf.append(t)
        flush()
    print(f"[chunk] {len(chunks)} chunks from {len(pages)} pages")
    return chunks


_model = None  # 埋め込みモデルの遅延ロード＆キャッシュ（呼び出しごとに再ロードしない）


def embed(texts, kind):
    # 埋め込みモデルは初回のみロードしてキャッシュ。e5系は "query: "/"passage: " のprefixが必須。
    global _model
    if _model is None:
        _model = SentenceTransformer(EMB_MODEL)
    prefix = "query: " if kind == "query" else "passage: "
    return _model.encode([prefix + t for t in texts], normalize_embeddings=True)


def index(vecs, chunks):
    # 埋め込みベクトルとメタ情報を index.npz に保存する。
    meta = [asdict(c) for c in chunks]
    np.savez(
        INDEX_PATH,
        vecs=np.asarray(vecs, dtype="float32"),
        meta=np.array(json.dumps(meta, ensure_ascii=False)),
    )
    print(f"[ingest] saved -> {INDEX_PATH}")


# ---- 検索 + 棄却 + 回答 ----------------------------------------------------


def retrieve(query):
    d = np.load(INDEX_PATH, allow_pickle=True)  # 索引(index.npz)を読み込む
    vecs, meta = d["vecs"], json.loads(str(d["meta"]))
    q = embed([query], "query")[0]
    scores = vecs @ q
    idx = np.argsort(-scores)[:TOP_K]
    return [(float(scores[i]), meta[i]) for i in idx]


def contextualize(query, hits):
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


def generate(prompt: str) -> str:
    """Anthropic Messages API でテキスト応答を返す。

    APIキーはコードに書かず os.environ["ANTHROPIC_API_KEY"] から読む（SDKが自動取得）。
    モデルは LLM_MODEL（環境変数 OTA_LLM_MODEL で上書き可）。
    """
    client = anthropic.Anthropic()  # ANTHROPIC_API_KEY を環境変数から取得
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
    hits = retrieve(query)
    top = hits[0][0] if hits else 0.0
    if top < SCORE_FLOOR:
        return {
            "abstain": True,
            "reason": "low_score",
            "top_score": top,
            "answer": "区の案内では確認できませんでした。ごみ減量推進課(03-5744-1628)へご確認ください。",
            "hits": hits,
        }
    answer = generate(
        contextualize(query, hits)
    )  # スタブ。未実装なら eval は retrieval のみ評価
    return {"abstain": False, "top_score": top, "answer": answer, "hits": hits}


# ---- 観測: ask パイプラインを goldenset 全問で一周し記録 (Phase 4) ------------


def observe_one(item):
    """1問を ask() に通し、観測レコード(dict)を返す。"""
    q = item["query"]
    res = ask(q)  # 実パイプライン（route→retrieve→棄却→生成）
    routed = route(q)  # 観測用に route 判定を別取得（ask 内部と同じ判定）
    hits = res.get("hits", [])
    top_k = [
        {
            "score": round(float(score), 4),
            "category": m.get("category"),
            "section": m.get("section"),
            "text_head": (m.get("text") or "")[:TEXT_HEAD],
        }
        for score, m in hits
    ]
    top_score = res.get("top_score")
    if top_score is None and top_k:
        top_score = top_k[0]["score"]
    return {
        "id": item["id"],
        "query": q,
        "should_abstain": item["should_abstain"],
        "expected_category": item["expected_category"],
        "routed": routed,
        "top_score": round(float(top_score), 4) if top_score is not None else None,
        "top_k": top_k,
        "generated_answer": res.get("answer"),
        "abstained": res.get("abstain"),
    }


def observe(goldenset_path="goldenset.json"):
    """goldenset 全問を ask に通し observation_log.jsonl / observation_summary.md を生成。"""
    goldenset = json.load(open(goldenset_path, encoding="utf-8"))
    records = []
    for item in goldenset:
        print(f"[observe] id={item['id']}: {item['query'][:28]} ...")
        records.append(observe_one(item))
    write_observation_log(records)
    write_observation_summary(records, SCORE_FLOOR, TOP_K)
    print(
        f"[observe] {len(records)} 問を観測 -> observation_log.jsonl / observation_summary.md"
    )


def update_index():
    # 取得→chunking→観察用ダンプ→埋め込み→索引保存 を一括実行し index.npz を更新する。
    pages = ingest()  # 取得（ingestion）
    chunks = chunk(pages)  # chunking
    dump_for_inspection(chunks, CATEGORIES, len(SOURCES))  # 観察用ダンプ＋category集計
    vecs = embed([c.text for c in chunks], "passage")
    index(vecs, chunks)


# ---- CLI -------------------------------------------------------------------
if __name__ == "__main__":
    load_dotenv()  # .env から ANTHROPIC_API_KEY を読む（ask/observe の generate 用。export 不要）
    cmd = sys.argv[1] if len(sys.argv) > 1 else "help"
    if cmd == "update-index":
        update_index()
    elif cmd == "ask":
        res = ask(sys.argv[2])
        print(json.dumps(res, ensure_ascii=False, indent=2))
    elif cmd == "observe":
        observe(sys.argv[2] if len(sys.argv) > 2 else "goldenset.json")
    else:
        print(__doc__)
