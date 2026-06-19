# rag-lab

RAG（検索拡張生成）を手を動かして学ぶための実験用リポジトリ。
検証・学習テーマごとにディレクトリを分け、それぞれで実際に RAG を動かしながら理解を深める。

## 構成

検証・学習テーマ（実験）ごとにディレクトリを分けている。各実験は自分用の
`rag.py`（同じ7段パイプライン）を持ち、自己完結して動く。共有物だけルートに置く。

```
minimal-rag/         最小RAGを素コードで7段体感する（トイ・コーパス）
real-corpus-rag/     実コーパスを収集・日本語化して検証する
pyproject.toml       依存定義（anthropic / requests / python-dotenv）※共有
uv.lock              uv が解決した依存のロックファイル ※共有
.env / .env.example  API キー（.env は gitignore 済）※共有
.venv/               uv が管理する仮想環境 ※共有
```

各実験の内部構成は、それぞれの README に記載している（下のリンク表）。

各実験の**目的・実行方法・観察ポイント**は、それぞれの README を参照:

| ディレクトリ | 目的 | README |
|---|---|---|
| `minimal-rag/` | 素コードでRAG7段を1本通し、中間データを目視して「データの流れ」を体感（学習） | [minimal-rag/README.md](minimal-rag/README.md) |
| `real-corpus-rag/` | raw大量コーパスを収集・日本語化して法律RAG(DRM/SAC)を検証 | [real-corpus-rag/README.md](real-corpus-rag/README.md) |

## セットアップ

依存と Python 環境は [uv](https://docs.astral.sh/uv/) で管理する。未導入なら:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

リポジトリ直下で一度同期すると、`.venv/` が作られ依存が入る（`uv.lock` も生成・更新される）:

```bash
uv sync
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

以降の実行は、リポジトリ内のどこからでも `uv run python minimal-rag/rag.py ...` の
ようにスクリプトをパス指定で叩く（`rag.py` は読み込み元を `__file__` 基準で解決するので
`cd` 不要）。`uv run` がルート共有の `.venv/` を自動で使う（手動の有効化も不要）。

> **セットアップはこのルート README だけで完結する**（`.venv/`・`.env` はルート共有）。
> ただし、例外的なセットアップが必要な場合、各実験ディレクトリの README に記載する。
> まずここで `uv sync` と `.env` の用意を済ませてから、各実験の README に進む。

> Anthropic は embeddings API を提供していないため、埋め込みだけ Voyage AI を使う。
> 生成は Anthropic 公式SDK + `claude-opus-4-8`。安く回したいときは各 `rag.py` の
> `GEN_MODEL` を `claude-haiku-4-5` に変える。

## 次に

セットアップが済んだら、各実験ディレクトリの README に進む。実行手順・観察ポイントは
そちらにまとまっている。

- [minimal-rag/README.md](minimal-rag/README.md) — 素コードでRAG7段を体感する（まずはここから）
- [real-corpus-rag/README.md](real-corpus-rag/README.md) — 実コーパスを収集・日本語化して検証する

## License

This repository is dual-licensed by file type:

- **Source code & config** — files whose extensions are listed in
  [LICENSE](LICENSE) (`.py`, `.ts`, `.tsx`, `.js`, `.jsx`, `.toml`, `.lock`,
  plus `.gitignore` and `.env.example`) — MIT License.
- **Everything else** — all documentation, corpus, data, and non-code assets,
  in any format, present or future (`*.md`, `*.txt`, `*.csv`, images, etc.) —
  © 2026 Yuta Igarashi (@yut496). All rights reserved. See
  [LICENSE-docs](LICENSE-docs). No permission is granted to use these materials
  as training data for machine learning or generative AI systems.
