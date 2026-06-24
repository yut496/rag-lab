"""Phase 5-B 単一変更: ナビ/関連リンク・チャンクを除去した v2 索引を生成する。

単一変更の純度を保つため、再 fetch せず既存 index.npz を「行フィルタ」して v2 を作る
（除去対象以外のチャンクのベクトルは v1 と byte 同一＝テキスト不変なら埋め込み不変）。
除去方法: 既知ナビテキストの section名パターン除外（task の「既知ナビテキストのパターン除外」）。

  .venv/bin/python build_index_v2.py
    index.npz -> index_v2.npz（v1 は保持）。除去前後のチャンク数と内訳を出力。
"""

import json

import numpy as np

V1_PATH = "index.npz"
V2_PATH = "index_v2.npz"

# <main> 内 <li> 経由で混入する「本文でないメニュー領域」の section（chunks_dump 分析で誤検出ゼロ）。
NAV_SECTIONS = {"関連リンク", "関連情報", "お問い合わせ", "家庭から出る資源とごみ"}


def main():
    d = np.load(V1_PATH, allow_pickle=True)
    vecs = d["vecs"]
    meta = json.loads(str(d["meta"]))
    assert len(meta) == len(vecs), "index.npz の vecs と meta の件数が不一致"

    keep_idx = [i for i, m in enumerate(meta) if m["section"] not in NAV_SECTIONS]
    removed = [m for m in meta if m["section"] in NAV_SECTIONS]

    vecs_v2 = np.asarray(vecs[keep_idx], dtype="float32")
    meta_v2 = [meta[i] for i in keep_idx]
    np.savez(V2_PATH, vecs=vecs_v2, meta=np.array(json.dumps(meta_v2, ensure_ascii=False)))

    # 除去前後のチャンク数
    print(f"[build_v2] v1 chunks={len(meta)} -> v2 chunks={len(meta_v2)} (removed={len(removed)})")

    # section 別・category 別の除去内訳
    by_sec, by_cat = {}, {}
    for m in removed:
        by_sec[m["section"]] = by_sec.get(m["section"], 0) + 1
        by_cat[m["category"]] = by_cat.get(m["category"], 0) + 1
    print("[build_v2] 除去 section 別:")
    for s in sorted(by_sec, key=lambda k: -by_sec[k]):
        print(f"    {s}: {by_sec[s]}")
    print("[build_v2] 除去 category 別:")
    for c in sorted(by_cat, key=lambda k: -by_cat[k]):
        print(f"    {c}: {by_cat[c]}")

    # 検証: v2 にナビ section が残っていないこと / vecs と meta の整合
    leftover = [m["section"] for m in meta_v2 if m["section"] in NAV_SECTIONS]
    assert len(meta_v2) == len(vecs_v2), "v2 の vecs と meta の件数が不一致"
    print(f"[build_v2] v2 に残存するナビ section: {leftover if leftover else 'なし（OK）'}")
    print(f"[build_v2] saved -> {V2_PATH}（v1 {V1_PATH} は保持）")


if __name__ == "__main__":
    main()
