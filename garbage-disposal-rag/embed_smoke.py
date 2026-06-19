"""埋め込みモデルの疎通確認スクリプト。

intfloat/multilingual-e5-base でダミー3文を埋め込み、正規化cosの類似度行列を出す。
期待: 「粗大ごみ ⇄ 大型ごみ」の類似度が「粗大ごみ ⇄ 可燃ごみの収集日」より高い。

実行:
  python embed_smoke.py
"""

import numpy as np
from sentence_transformers import SentenceTransformer

EMB_MODEL = "intfloat/multilingual-e5-base"

# e5 のプレフィックス: ここは「文どうしの対称的な意味類似」を見たいので、
# 全文を query: で統一する（retrieval の query/passage 非対称とは用途が異なる）。
SENTENCES = [
    "粗大ごみの出し方",
    "大型ごみの申込方法",
    "可燃ごみの収集日",
]


def main():
    model = SentenceTransformer(EMB_MODEL)
    vecs = model.encode(
        [f"query: {s}" for s in SENTENCES],
        normalize_embeddings=True,
    )
    vecs = np.asarray(vecs)
    sim = vecs @ vecs.T   # 正規化済みなので内積 = cos類似度

    print(f"model: {EMB_MODEL}")
    print("文:")
    for i, s in enumerate(SENTENCES):
        print(f"  [{i}] {s}")
    print("\ncos類似度行列:")
    header = "      " + " ".join(f"[{j}]   " for j in range(len(SENTENCES)))
    print(header)
    for i, row in enumerate(sim):
        print(f"  [{i}] " + " ".join(f"{v:.3f}" for v in row))

    s_sodai_oogata = float(sim[0][1])   # 粗大ごみ ⇄ 大型ごみ
    s_sodai_kanen = float(sim[0][2])    # 粗大ごみ ⇄ 可燃ごみの収集日
    print(f"\n粗大⇄大型       = {s_sodai_oogata:.3f}")
    print(f"粗大⇄可燃収集日 = {s_sodai_kanen:.3f}")
    ok = s_sodai_oogata > s_sodai_kanen
    print(f"\n期待（粗大⇄大型 > 粗大⇄可燃収集日）: {'PASS' if ok else 'FAIL'}")


if __name__ == "__main__":
    main()
