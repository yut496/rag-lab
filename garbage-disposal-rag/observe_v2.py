"""Phase 5-B: v2 索引（ナビ除去済み）で goldenset を観測する。

要ネットワーク/モデル（埋め込み + 生成）。サンドボックス不可のため**ユーザーが実行**する。

単一変更の遵守:
- retrieve を v2 索引へ向けるためだけに **実行時に rag.INDEX_PATH を差し替える**（rag.py のソースは無変更）。
- route / SCORE_FLOOR / プロンプト / 埋め込みモデル / チャンク分割は rag のまま（一切触らない）。
- goldenset 不変。出力は observation_log_v2.jsonl（v1 の observation_log.jsonl は保持）。

  .venv/bin/python observe_v2.py
    -> observation_log_v2.jsonl（15問）
"""

import json

import helper
import rag
from dotenv import load_dotenv

V2_INDEX = "index_v2.npz"
OUT_LOG = "observation_log_v2.jsonl"
GOLDENSET = "goldenset.json"


def main():
    # generate() の Anthropic 認証用に .env を読む（rag.py は __main__ 内で load_dotenv するが、
    # ここは import 経由で observe_one を呼ぶため __main__ を通らない → 自前で読む）。
    load_dotenv()

    # retrieve が参照する索引パスだけ v2 に差し替える（ソース変更なし・論理は不変）。
    rag.INDEX_PATH = V2_INDEX
    print(f"[observe_v2] INDEX_PATH -> {rag.INDEX_PATH}")

    goldenset = json.load(open(GOLDENSET, encoding="utf-8"))
    records = []
    for item in goldenset:
        print(f"[observe_v2] id={item['id']}: {item['query'][:28]} ...")
        records.append(rag.observe_one(item))  # v1 と同一の実パイプライン（索引のみ v2）

    helper.write_observation_log(records, OUT_LOG)
    print(f"[observe_v2] {len(records)} 問を観測 -> {OUT_LOG}")


if __name__ == "__main__":
    main()
