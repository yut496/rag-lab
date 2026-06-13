"""
最小RAGパイプラインを「素のコード」で1本通すスクリプト。

目的は性能でも汎用性でもなく、RAGの7段を上から下へ読みながら
「データがどの段でどう変形するか」を目で見て体感すること。
各段の境界で中間データを print する。フレームワーク（LangChain等）は使わない。

  ingestion → chunking → embedding → index → retrieval → context組み立て → generation

前提:
  - 生成   = Claude（Anthropic Messages API、claude-opus-4-8）→ ANTHROPIC_API_KEY
  - 埋め込み = Voyage AI（Anthropicは embeddings API を提供しないため）→ VOYAGE_API_KEY
  - 検索   = 純Pythonの手書きコサイン類似度（numpyすら使わない）

使い方:
  cp .env.example .env  でキーを記入（または環境変数で export）:
    ANTHROPIC_API_KEY=...   # https://console.anthropic.com
    VOYAGE_API_KEY=...      # https://dashboard.voyageai.com
  python rag.py "How many termites can the azure pangolin eat in a night?"
"""

import math
import os
import sys
from pathlib import Path

import requests
import anthropic
from dotenv import load_dotenv

# ---- チューニングパラメータ（ここを変えて各段の挙動を体感する） --------------
DOCS_DIR = Path(__file__).parent / "docs"
CHUNK_SIZE = 300        # チャンク1個の文字数。小さくすると正解が途切れやすくなる
CHUNK_OVERLAP = 50      # 隣り合うチャンクの重なり。境界での文脈断裂を防ぐ
TOP_K = 3               # retrievalで上位何件を文脈に渡すか
EMBED_MODEL = "voyage-4-lite"   # 他に voyage-4 / voyage-4-large など
GEN_MODEL = "claude-opus-4-8"   # 安く回したいなら "claude-haiku-4-5"
VOYAGE_URL = "https://api.voyageai.com/v1/embeddings"

DEFAULT_QUESTION = "How many termites can the azure pangolin eat in a night?"


def banner(stage: str) -> None:
    print("\n" + "=" * 70)
    print(f"■ {stage}")
    print("=" * 70)


# ============================================================================
# ① ingestion — ディスク上の生テキストを読み込む
#    現実ではPDF/HTML/DBなど多様。ここは docs/*.txt を素直に読むだけ。
# ============================================================================
def ingest():
    docs = []
    for path in sorted(DOCS_DIR.glob("*.txt")):
        docs.append((path.name, path.read_text(encoding="utf-8")))
    banner("① ingestion")
    print(f"{len(docs)} 件の文書を読み込み:")
    for doc_id, text in docs:
        print(f"  - {doc_id}  ({len(text)} 文字)")
    return docs


# ============================================================================
# ② chunking — 文書を検索単位（チャンク）に割る
#    RAGで最も地味で最も事故る段。固定長＋オーバーラップの素朴な実装。
#    分割が雑だと「正解が2チャンクに割れる/文の途中で切れる」が起きる。
#    → CHUNK_SIZE を小さくして再実行すると体感できる。
# ============================================================================
def chunk_one(text: str, size: int, overlap: int):
    chunks = []
    start = 0
    while start < len(text):
        end = start + size
        chunks.append(text[start:end])
        if end >= len(text):
            break
        start = end - overlap   # overlap 分だけ戻して次のチャンクを取る
    return chunks


def chunk(docs):
    # chunk = {"text": ..., "source": ...}。source で「どの文書由来か」を保持。
    chunks = []
    for doc_id, text in docs:
        for piece in chunk_one(text, CHUNK_SIZE, CHUNK_OVERLAP):
            chunks.append({"text": piece, "source": doc_id})
    banner("② chunking")
    print(f"CHUNK_SIZE={CHUNK_SIZE}, OVERLAP={CHUNK_OVERLAP} → {len(chunks)} チャンク")
    print("先頭チャンクの例:")
    print(f"  [{chunks[0]['source']}] {chunks[0]['text']!r}")
    return chunks


# ============================================================================
# ③ embedding — テキストを「意味の座標（ベクトル）」に変換する
#    Voyage REST を requests で直接叩き、生の float リストを取り出す。
#    input_type を document/query で分けるのが非対称埋め込み:
#    「文書として埋める」のと「問いとして埋める」で最適化が異なる。
# ============================================================================
def embed(texts, input_type: str):
    key = os.environ.get("VOYAGE_API_KEY")
    if not key:
        sys.exit("VOYAGE_API_KEY が未設定です。.env に書くか export してください。")
    resp = requests.post(
        VOYAGE_URL,
        headers={"Authorization": f"Bearer {key}"},
        json={"input": texts, "model": EMBED_MODEL, "input_type": input_type},
        timeout=30,
    )
    if resp.status_code != 200:
        sys.exit(f"Voyage API エラー {resp.status_code}: {resp.text}")
    data = resp.json()["data"]
    # data は index 順とは限らないので index で並べ直す
    data.sort(key=lambda d: d["index"])
    return [d["embedding"] for d in data]


def embed_chunks(chunks):
    vectors = embed([c["text"] for c in chunks], input_type="document")
    banner("③ embedding")
    dim = len(vectors[0])
    print(f"{len(vectors)} 本のベクトルを取得。各ベクトルは {dim} 次元。")
    print(f"先頭チャンクのベクトル先頭5要素: {vectors[0][:5]}")
    return vectors


# ============================================================================
# ④ index — 検索対象を保持する“索引”
#    本質はこの (text, source, vector) のリストそのもの。
#    実運用のANN（HNSW/IVF）は「総当たりを高速化する最適化」にすぎない。
#    トイ規模なので総当たりで十分。
# ============================================================================
def build_index(chunks, vectors):
    index = [
        {"text": c["text"], "source": c["source"], "vector": v}
        for c, v in zip(chunks, vectors)
    ]
    banner("④ index")
    print(f"メモリ上の索引 = {len(index)} 件 × {len(vectors[0])} 次元のリスト")
    return index


# ============================================================================
# ⑤ retrieval — 問いを埋め込み、近いチャンクを引く
#    コサイン類似度を純Pythonで手書き（内積/ノルムのループが見える）。
#    retrievalは“局所的に上位数件を引くだけ”＝母集団全体は見ていない、を体感。
# ============================================================================
def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb) if na and nb else 0.0


def retrieve(index, query: str):
    q_vec = embed([query], input_type="query")[0]   # 問いは query として埋める
    scored = [
        {"score": cosine(q_vec, item["vector"]), **item}
        for item in index
    ]
    scored.sort(key=lambda s: s["score"], reverse=True)

    banner("⑤ retrieval")
    print(f"問い: {query!r}")
    print("全チャンクのコサイン類似度（降順）:")
    for s in scored:
        print(f"  {s['score']:.4f}  [{s['source']}] {s['text'][:50]!r}...")
    top = scored[:TOP_K]
    print(f"\n→ 上位 {TOP_K} 件を文脈に採用。")
    return top


# ============================================================================
# ⑥ context組み立て — 引いたチャンクを1つの文字列にまとめる
#    LLMに実際に渡す文字列そのものを表示する。出典ラベル付き。
# ============================================================================
def build_context(top):
    blocks = [f"[Source: {s['source']}]\n{s['text']}" for s in top]
    context = "\n\n---\n\n".join(blocks)
    banner("⑥ context組み立て（LLMに渡す文字列そのもの）")
    print(context)
    return context


# ============================================================================
# ⑦ generation — 文脈＋問いをClaudeに渡して回答させる
#    system で「文脈だけを根拠に。無ければ分からないと言う」と縛る。
#    → コーパス外を質問すると grounding（捏造抑制）の挙動を体感できる。
# ============================================================================
def generate(context: str, question: str):
    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("ANTHROPIC_API_KEY が未設定です。.env に書くか export してください。")
    client = anthropic.Anthropic()
    system = (
        "You answer strictly using the provided context. "
        "If the answer is not in the context, say you don't know. "
        "Do not use outside knowledge."
    )
    user = f"Context:\n{context}\n\nQuestion: {question}"
    resp = client.messages.create(
        model=GEN_MODEL,
        max_tokens=1024,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    answer = "".join(b.text for b in resp.content if b.type == "text")
    banner("⑦ generation")
    print(f"問い: {question}")
    print(f"\n回答:\n{answer}")
    return answer


def main():
    # .env から API キーを読み込む（既存の環境変数があればそちらが優先）
    load_dotenv()

    question = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_QUESTION

    docs = ingest()
    chunks = chunk(docs)
    vectors = embed_chunks(chunks)
    index = build_index(chunks, vectors)
    top = retrieve(index, question)
    context = build_context(top)
    generate(context, question)


if __name__ == "__main__":
    main()
