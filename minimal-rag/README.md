# minimal-rag — 最小RAGを素コードで体感する

**目的:** フレームワーク（LangChain / LlamaIndex）を使わず、素の Python で RAG の7段
— ingestion → chunking → embedding → index → retrieval → context組み立て → generation —
を1本通す。各段の境界で中間データを print し、「データがどの段でどう変形するか」を目視して
体感するための**学習用**。検索のコサイン類似度も numpy を使わず手書きで、隠蔽を避けている。

## 構成

```
rag.py     7段を順に実行する単一スクリプト
corpus/    トイ・コーパス（英語）。事実が離散的で一意な3文書
  helix3000.txt      架空のエスプレッソマシンの仕様
  mistvale.txt       架空の山間の町の事実
  azure_pangolin.txt 架空の動物の事実
NOTE.md    学習全体の履歴（疑問と回答、メモ、ゴールのアウトプット）
```

**前提**:

- 生成 = Claude（Anthropic Messages API, claude-opus-4-8）→ `ANTHROPIC_API_KEY`
- 埋め込み = Voyage AI（Anthropic は embeddings API を提供しないため）→ `VOYAGE_API_KEY`
- 検索 = 純Pythonの手書きコサイン類似度

> **NOTE.md のルール**: 記入はすべて自分の言葉で書く（コピペ・引用・AI生成ではなく、噛み砕いて説明する）。

トイ・コーパスはあえて**架空の事実**にしてある。LLM の事前知識と混ざらないので、
「文脈だけを根拠に答えているか（grounding）」を切り分けて観察できる。

## ゴール

このトイRAGで達成したいことを先に決めておく。rag.py を走らせ、各ゴールを確認する。

- **RAGの全体フローと各処理でデータがどう変形するかを説明できる**:
- **chunking が retrieval を揺らすことを体感する**:
  `CHUNK_SIZE` を 80 など小さくして再実行し、正解が複数チャンクに割れ・文の途中で切れ、retrievalスコアや回答がどう変わるか。
- **input_type の非対称埋め込みを理解する**:
  embedingは文書を `document`、retrievalは問いを `query` で埋める。
  同じテキストでも「文書として」と「問いとして」で埋め方が変わる。
- **retrieval が正解チャンクを上位に引けることを確認する**:
  retrievalのコサイン類似度ランキングで、答えを含むチャンクが上位に来るか。来なければ「検索の失敗」。
- **grounding（捏造抑制）が働くことを確認する**:
  コーパスに無い質問
  （`cd minimal-rag && uv run python rag.py "What is the capital of France?"`）で、無関係なチャンクが
  引かれ Claude が「分からない」と答える。

## 実行

この実験は**独立した uv 環境**を持つ。初回はこのディレクトリで一度同期する
（`.venv/` と `uv.lock` が作られる。`.env` はリポジトリ直下のものを共有する）:

```bash
cd minimal-rag
uv sync
```

以降はこのディレクトリの中で実行する。`rag.py` は読み込み元やログを自分の位置
（`__file__`）基準で解決するので、コーパス等のパスは `cd` 後でもそのまま通る:

```bash
# コーパス内の質問（答えは corpus/azure_pangolin.txt にある）
uv run python rag.py "How many termites can the azure pangolin eat in a night?"

# 引数なしならデフォルト質問が使われる
uv run python rag.py
```

各段が順に表示され、最後に Claude の回答が出る。
