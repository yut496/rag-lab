# basic-rag-system

最小RAGパイプラインを「素のコード」で1本通すための学習用リポジトリ。
RAGの7段 — ingestion → chunking → embedding → index → retrieval → context組み立て → generation —
を上から下へ読みながら、各段の境界で中間データを目視し「データの流れ」を体感する。

フレームワーク（LangChain / LlamaIndex）は使わない。検索のコサイン類似度も
numpyを使わず純Pythonで手書きしてある（隠蔽を避け、何が起きているか分かるように）。

## 構成

```
docs/              トイ・コーパス（英語）。事実が離散的で一意な3文書
  helix3000.txt      架空のエスプレッソマシンの仕様
  mistvale.txt       架空の山間の町の事実
  azure_pangolin.txt 架空の動物の事実
rag.py             7段を順に実行する単一スクリプト
requirements.txt   依存（anthropic, requests のみ）
```

## セットアップ

最近のmacOS/Pythonは `pip install` を直接拒否する（PEP 668）。venvを使う:

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

APIキーは `.env` から読む（python-dotenv）。テンプレートをコピーして値を記入:

```bash
cp .env.example .env
# .env を編集:
#   ANTHROPIC_API_KEY=...   # 生成（Claude）用    https://console.anthropic.com
#   VOYAGE_API_KEY=...      # 埋め込み（Voyage）用  https://dashboard.voyageai.com
```

`.env` は `.gitignore` 済み（コミットされない）。

> ⚠️ **`ANTHROPIC_API_KEY` を `export` しないこと（必ず `.env` を使う）。**
> Claude Code は起動時にシェルの `ANTHROPIC_API_KEY` を拾い、その鍵で課金され得る。
> rag.py 用に同じシェルで `export ANTHROPIC_API_KEY=...` すると、**Claude Code での作業が
> 意図せず自分のAPI契約（従量課金）を消費してしまう**事故が起きる。
> `.env` は Claude Code に読まれないため、鍵は `.env` に置けばこの混線を避けられる。
> （技術的には環境変数が `.env` より優先されるが、上記の理由で `export` は使わない。）

以降の実行は `.venv/bin/python rag.py ...`（または `source .venv/bin/activate` 後に `python rag.py ...`）。

> Anthropic は embeddings API を提供していないため、埋め込みだけ Voyage AI を使う。
> 生成は Anthropic 公式SDK + `claude-opus-4-8`。安く回したいときは `rag.py` の
> `GEN_MODEL` を `claude-haiku-4-5` に変える。

## 実行

```bash
# コーパス内の質問（答えは azure_pangolin.txt にある）
python rag.py "How many termites can the azure pangolin eat in a night?"

# 引数なしならデフォルト質問が使われる
python rag.py
```

各段が順に表示され、最後に Claude の回答が出る。

## 観察ガイド（ここからが学び）

- **retrievalのランキングを見る**: ⑤で全チャンクのコサイン類似度が降順表示される。
  答えを含むチャンクが上位に来ているか？ 来ていなければ「検索の失敗」。
- **grounding（捏造抑制）を試す**: コーパスに無い質問を投げる。
  ```bash
  python rag.py "What is the capital of France?"
  ```
  検索は無関係なチャンクを引き、Claudeは「分からない」と答えるはず（system promptで縛っている）。
- **chunkingの事故を体感する**: `rag.py` の `CHUNK_SIZE` を 80 など小さくして再実行。
  正解が複数チャンクに割れたり文の途中で切れたりして、retrievalスコアや回答がどう変わるか観察。
- **input_type の非対称埋め込み**: ③では文書を `document`、⑤では問いを `query` として埋めている。
  同じテキストでも「文書として」と「問いとして」で埋め方が変わる点に注目。
