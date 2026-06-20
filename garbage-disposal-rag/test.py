"""疎通確認スクリプト（smoke tests）— 埋め込みと LLM をまとめて確認する。

  uv run python test.py          # 両方
  uv run python test.py embed    # 埋め込みのみ
  uv run python test.py llm      # LLM のみ
"""

import sys

import numpy as np
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

from rag import LLM_MODEL, call_llm

EMB_MODEL = "intfloat/multilingual-e5-base"
SMOKE_SENTENCES = [
    "粗大ごみの出し方",
    "大型ごみの申込方法",
    "可燃ごみの収集日",
]


def embed_smoke():
    """e5 でダミー3文を埋め込み、cos類似度行列を出して期待関係を確認する。

    期待: 「粗大ごみ ⇄ 大型ごみ」 > 「粗大ごみ ⇄ 可燃ごみの収集日」。
    """
    model = SentenceTransformer(EMB_MODEL)
    # e5 のプレフィックスは、文どうしの対称的な意味類似を見るため全文 query: で統一。
    vecs = np.asarray(model.encode(
        [f"query: {s}" for s in SMOKE_SENTENCES],
        normalize_embeddings=True,
    ))
    sim = vecs @ vecs.T   # 正規化済みなので内積 = cos類似度

    print(f"[embed_smoke] model: {EMB_MODEL}")
    for i, s in enumerate(SMOKE_SENTENCES):
        print(f"  [{i}] {s}")
    print("  cos類似度行列:")
    for row in sim:
        print("    " + " ".join(f"{v:.3f}" for v in row))
    s12, s13 = float(sim[0][1]), float(sim[0][2])
    print(f"  粗大⇄大型 = {s12:.3f} / 粗大⇄可燃収集日 = {s13:.3f}")
    ok = s12 > s13
    print(f"  期待（粗大⇄大型 > 粗大⇄可燃収集日）: {'PASS' if ok else 'FAIL'}")
    return ok


def llm_smoke():
    """call_llm に "test" を渡し、非空のテキスト応答が返ることを確認する。

    キーは export せず、リポジトリ直下の .env から読む（python-dotenv）。
    """
    load_dotenv()   # cwd〜親をたどり rag-lab/.env を読む（os.environ に注入）
    out = call_llm("test")
    print(f"[llm_smoke] model: {LLM_MODEL}")
    print(f"[llm_smoke] response: {out!r}")
    ok = isinstance(out, str) and bool(out.strip())
    print(f"  期待（非空のテキスト応答）: {'PASS' if ok else 'FAIL'}")
    return ok


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    results = {}
    if which in ("all", "embed"):
        results["embed_smoke"] = embed_smoke()
    if which in ("all", "llm"):
        results["llm_smoke"] = llm_smoke()
    if not results:
        sys.exit(f"unknown target: {which!r}（embed / llm / 省略=both）")
    print("\n=== summary ===")
    for name, ok in results.items():
        print(f"  {name}: {'PASS' if ok else 'FAIL'}")
    sys.exit(0 if all(results.values()) else 1)


if __name__ == "__main__":
    main()
