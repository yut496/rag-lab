"""最小RAGパイプラインを「素のコード」で1本通すスクリプト。"""

import math
import os
import sys
from pathlib import Path

import requests
import anthropic
from dotenv import load_dotenv

# ---- チューニングパラメータ（ここを変えて各段の挙動を体感する） --------------
DOCS_DIR = Path(__file__).parent / "corpus"   # ingestion の読み込み元ディレクトリ
CHUNK_SIZE = 300        # チャンク1個の文字数。小さくすると正解が途切れやすくなる
# CHUNK_SIZE = 80
CHUNK_OVERLAP = 50      # 隣り合うチャンクの重なり。境界での文脈断裂を防ぐ
TOP_K = 3               # retrievalで上位何件を文脈に渡すか
EMBED_MODEL = "voyage-4-lite"   # 他に voyage-4 / voyage-4-large など
GEN_MODEL = "claude-opus-4-8"   # 安く回したいなら "claude-haiku-4-5"
VOYAGE_URL = "https://api.voyageai.com/v1/embeddings"

DEFAULT_QUESTION = "How many termites can the azure pangolin eat in a night?"


def banner(stage: str) -> None:
    """各段の区切りを標準出力に印字する。"""
    print("\n" + "=" * 70)
    print(f"■ {stage}")
    print("=" * 70)


def ingest(docs_dir: Path):
    """① ingestion — ディスク上の生テキストを読み込む。

    現実ではPDF/HTML/DBなど多様。ここは corpus/*.txt を素直に読むだけ。
    """
    docs = []
    for path in sorted(docs_dir.glob("*.txt")):
        docs.append((path.name, path.read_text(encoding="utf-8")))
    if not docs:
        sys.exit(f"{docs_dir} に *.txt が見つかりません。コーパスの場所を確認してください。")
    banner("① ingestion")
    print(f"読み込み元: {docs_dir}")
    print(f"{len(docs)} 件の文書を読み込み:")
    for doc_id, text in docs:
        print(f"  - {doc_id}  ({len(text)} 文字)")
    return docs


def chunk_one(text: str, size: int, overlap: int):
    """固定長＋オーバーラップで1文書を分割する。"""
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
    """② chunking — 文書を検索単位（チャンク）に割る。

    RAGで最も地味で最も事故る段。固定長＋オーバーラップの素朴な実装。
    分割が雑だと「正解が2チャンクに割れる/文の途中で切れる」が起きる
    （CHUNK_SIZE を小さくして再実行すると体感できる）。
    各 chunk は {"text": ..., "source": ...}。source で「どの文書由来か」を保持。
    """
    chunks = []
    for doc_id, text in docs:
        for piece in chunk_one(text, CHUNK_SIZE, CHUNK_OVERLAP):
            chunks.append({"text": piece, "source": doc_id})
    banner("② chunking")
    print(f"CHUNK_SIZE={CHUNK_SIZE}, OVERLAP={CHUNK_OVERLAP} → {len(chunks)} チャンク")
    print("先頭チャンクの例:")
    print(f"  [{chunks[0]['source']}] {chunks[0]['text']!r}")
    return chunks


def embed(texts, input_type: str):
    """③ embedding — テキストを「意味の座標（ベクトル）」に変換する。

    Voyage REST を requests で直接叩き、生の float リストを取り出す。全チャンクを
    1リクエストで送る。input_type を document/query で分けるのが非対称埋め込み:
    「文書として埋める」のと「問いとして埋める」で最適化が異なる。
    """
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
    data.sort(key=lambda d: d["index"])   # data は index 順とは限らないので並べ直す
    return [d["embedding"] for d in data]


def embed_chunks(chunks):
    """チャンク群を document として埋め込み、③ embedding の中間データを表示する。"""
    vectors = embed([c["text"] for c in chunks], input_type="document")
    banner("③ embedding")
    dim = len(vectors[0])
    print(f"{len(vectors)} 本のベクトルを取得。各ベクトルは {dim} 次元。")
    print(f"先頭チャンクのベクトル先頭5要素: {vectors[0][:5]}")
    return vectors


def build_index(chunks, vectors):
    """④ index — 検索対象を保持する“索引”を作る。

    本質はこの (text, source, vector) のリストそのもの。
    実運用のANN（HNSW/IVF）は「総当たりを高速化する最適化」にすぎない。
    トイ規模なので総当たりで十分。
    """
    index = [
        {"text": c["text"], "source": c["source"], "vector": v}
        for c, v in zip(chunks, vectors)
    ]
    banner("④ index")
    print(f"メモリ上の索引 = {len(index)} 件 × {len(vectors[0])} 次元のリスト")
    return index


def cosine(a, b):
    """2ベクトルのコサイン類似度を純Pythonで計算する（内積/ノルムのループが見える）。"""
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb) if na and nb else 0.0


def retrieve(index, query: str):
    """⑤ retrieval — 問いを埋め込み、近いチャンクを引く。

    retrievalは“局所的に上位数件を引くだけ”＝母集団全体は見ていない、を体感する。
    """
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


def build_context(top):
    """⑥ context組み立て — 引いたチャンクを1つの文字列にまとめる。

    LLMに実際に渡す文字列そのものを表示する。出典ラベル付き。
    """
    blocks = [f"[Source: {s['source']}]\n{s['text']}" for s in top]
    context = "\n\n---\n\n".join(blocks)
    banner("⑥ context組み立て（LLMに渡す文字列そのもの）")
    print(context)
    return context


def generate(context: str, question: str):
    """⑦ generation — 文脈＋問いをClaudeに渡して回答させる。

    system で「文脈だけを根拠に。無ければ分からないと言う。使った出典を明記する」と縛る。
    → コーパス外を質問すると grounding（捏造抑制）の挙動を体感できる。出典明記で根拠も辿れる。
    """
    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("ANTHROPIC_API_KEY が未設定です。.env に書くか export してください。")
    client = anthropic.Anthropic()
    system = (
        "You answer strictly using the provided context. "
        "If the answer is not in the context, say you don't know. "
        "Do not use outside knowledge. "
        "Cite the source label (e.g. [Source: foo.txt]) you used in your answer."
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
    """7段を順に1本通す。第1引数があれば問い、無ければデフォルトの質問を使う。"""
    # .env から API キーを読み込む（既存の環境変数があればそちらが優先）
    load_dotenv()

    question = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_QUESTION

    docs = ingest(DOCS_DIR)
    chunks = chunk(docs)
    vectors = embed_chunks(chunks)
    index = build_index(chunks, vectors)
    top = retrieve(index, question)
    context = build_context(top)
    generate(context, question)


if __name__ == "__main__":
    main()
