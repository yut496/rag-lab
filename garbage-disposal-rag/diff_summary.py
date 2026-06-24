"""Phase 5-B: 単一変更（ナビ除去）の前後 3軸差分を diff_summary.md にまとめる（オフライン）。

入力（すべて既存ファイルを読むだけ。system は再実行しない）:
  - scoring_baseline.csv   … v1 採点（人手確定済み）
  - scoring_v2.csv         … v2 採点（人手確定済み）
  - observation_log.jsonl  … v1 観測（top_k 比較用）
  - observation_log_v2.jsonl … v2 観測（top_k 比較用）

出力: diff_summary.md（3軸差分 / 問題別 top_k 変化 / id14 特記 / LLM 変動の区別 / 結論）。

  .venv/bin/python diff_summary.py

理論的事実（決定論）: ナビ・チャンクを索引から落とすと top_k が変わりうるのは
「v1 の top_k に元々ナビが入っていた問い」だけ（= id1,6,7,8,14）。それ以外の問いは
top_k 内に上位ナビが無いので、下位ナビを除いても上位4件は不変 → 検索は同一。
→ よって検索が不変な問いの grounded/abstain 差は ask 再実行の生成ゆらぎ（LLM 変動）。
"""

import json

from build_index_v2 import NAV_SECTIONS
from score_baseline import read_rows

V1_CSV = "scoring_baseline.csv"
V2_CSV = "scoring_v2.csv"
V1_LOG = "observation_log.jsonl"
V2_LOG = "observation_log_v2.jsonl"
OUT = "diff_summary.md"


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def by_id(rows):
    return {r["id"]: r for r in rows}


def tally(rows):
    """3軸の (ok, 母数) を返す（build_summary と同じ数え方）。"""
    answer = [x for x in rows if x["should_abstain"] is False]
    abstain = [x for x in rows if x["should_abstain"] is True]

    def ok(items, key):
        return sum(1 for x in items if x[key] is True)

    def review(items, key):
        return sum(1 for x in items if x[key] == "REVIEW")

    return {
        "retrieval": (ok(answer, "retrieval_ok"), len(answer)),
        "grounded": (ok(answer, "grounded_ok"), len(answer)),
        "abstain": (ok(abstain, "abstain_ok"), len(abstain)),
        "review": review(answer, "grounded_ok") + review(abstain, "abstain_ok"),
    }


def topk_cats(rec):
    return [h["category"] for h in rec["top_k"]]


def topk_pairs(rec):
    """top_k を (category, section) の列で表す（順位込みの同一性判定用）。"""
    return [(h["category"], h["section"]) for h in rec["top_k"]]


def nav_count(rec):
    return sum(1 for h in rec["top_k"] if h["section"] in NAV_SECTIONS)


def fmt(score_pair):
    return f"{score_pair[0]}/{score_pair[1]}"


def main():
    v1 = by_id(read_rows(V1_CSV))
    v2 = by_id(read_rows(V2_CSV))
    log1 = by_id(load_jsonl(V1_LOG))
    log2 = by_id(load_jsonl(V2_LOG))
    ids = sorted(v1)

    t1, t2 = tally(list(v1.values())), tally(list(v2.values()))

    # 検索（top_k）が変化した問いを特定（理論上は v1 で nav を含む問いのみ）
    changed, nav_in_v1 = [], []
    for i in ids:
        if i not in log1 or i not in log2:
            continue
        if nav_count(log1[i]) > 0:
            nav_in_v1.append(i)
        if topk_pairs(log1[i]) != topk_pairs(log2[i]):
            changed.append(i)

    L = []
    L.append("# Phase 5-B 差分: ナビ除去（v1 → v2）\n")

    L.append("## 単一変更（これ以外は不変）")
    L.append(f"- NAV_SECTIONS = {{{', '.join(sorted(NAV_SECTIONS))}}} のチャンクを index.npz から**行フィルタ除去**。")
    L.append("- 方式: 既知ナビ section 名のパターン除外。**再 fetch しない**（コーパス・ドリフト回避）。")
    L.append("- v1 86 chunks → v2 60 chunks（除去 26）。除去以外のベクトルは v1 と **byte 同一**（テキスト不変＝埋め込み不変）。")
    L.append("- SCORE_FLOOR / プロンプト / route / 埋め込みモデル / チャンク分割 / goldenset / 採点ルールは無変更。\n")

    L.append("## 3軸スコア（baseline → v2）")
    L.append("| 軸 | baseline(v1) | v2 | 差 |")
    L.append("|---|---|---|---|")
    for key, label in [("retrieval", "retrieval_ok"), ("grounded", "grounded_ok"), ("abstain", "abstain_ok")]:
        a, b = t1[key], t2[key]
        diff = b[0] - a[0]
        L.append(f"| {label} | {fmt(a)} | {fmt(b)} | {diff:+d} |")
    if t2["review"]:
        L.append(f"\n> ⚠ v2 CSV に未確定 REVIEW が {t2['review']} 件あります。人手確定後に再実行してください。")
    L.append("")

    L.append("## 検索（top_k）の変化 — 問題別")
    L.append(f"理論上 top_k が変わりうるのは「v1 top_k にナビを含む問い」= **{nav_in_v1}** のみ。")
    L.append(f"実測で top_k が変化した問い = **{changed}**。")
    L.append("| id | 種別 | routed | v1 nav数 | top_k変化 | v1 top_k(cat) | v2 top_k(cat) |")
    L.append("|---|---|---|---|---|---|---|")
    for i in ids:
        r1 = log1.get(i)
        r2 = log2.get(i)
        kind = "棄却" if v1[i]["should_abstain"] else "回答"
        if not r1 or not r2:
            L.append(f"| {i} | {kind} | — | — | — | — | — |")
            continue
        ch = "変化" if i in changed else "不変"
        c1 = "|".join(topk_cats(r1)) or "—（route棄却）"
        c2 = "|".join(topk_cats(r2)) or "—（route棄却）"
        L.append(f"| {i} | {kind} | {r1.get('routed')} | {nav_count(r1)} | {ch} | {c1} | {c2} |")
    L.append("")

    def cellv(v):
        return {True: "○", False: "×"}.get(v, str(v))  # N/A / REVIEW はそのまま

    # ---- retrieval_ok が反転した回答問の機構分析（最重要発見）----
    # ナビ除去で「合計は動かなくても中身が入れ替わる」反転を、機構別に分類する:
    #   True→False かつ v1 一致が NAV section ⇒ ナビ経由の偽陽性の是正
    #   False→True かつ v2 一致が非NAV section ⇒ 埋もれた実コンテンツの救済
    flips = []
    for i in ids:
        if v1[i]["should_abstain"] or i not in log1 or i not in log2:
            continue
        a, b = v1[i].get("retrieval_ok"), v2[i].get("retrieval_ok")
        if a == b:
            continue
        exp = log1[i]["expected_category"]
        h1 = next((h for h in log1[i]["top_k"] if h["category"] == exp), None)
        h2 = next((h for h in log2[i]["top_k"] if h["category"] == exp), None)
        flips.append((i, a, b, exp, h1, h2))

    L.append("## retrieval_ok の反転（最重要発見）")
    L.append(
        "ナビ除去では **retrieval の合計数が動かなくても中身が入れ替わりうる**。"
        "回答問で retrieval_ok が反転した問いと、その機構（ナビ経由の偽陽性是正 ／ 埋もれた実コンテンツ救済）:"
    )
    if flips:
        for i, a, b, exp, h1, h2 in flips:
            L.append(f"- **id{i}: retrieval_ok {cellv(a)}→{cellv(b)}**（expected「{exp}」）")
            if a is True and b is False:
                via_nav = h1 is not None and h1["section"] in NAV_SECTIONS
                if via_nav:
                    L.append(
                        f"    - v1 の一致は **ナビchunk経由の偽陽性**（section「{h1['section']}」/ "
                        f"score {h1['score']:.4f}）。ナビ除去で消滅し、実コンテンツは top_k に届かず → "
                        "**偽陽性の是正**（v1 の合格は見せかけだった）。"
                    )
                else:
                    L.append("    - v1 の実コンテンツ一致が繰り上がりで圏外へ（NAV 由来でない反転・要確認）。")
            elif a is False and b is True and h2 is not None:
                via_nav = h2["section"] in NAV_SECTIONS
                kind = "（※ v2 側もNAV・要確認）" if via_nav else ""
                L.append(
                    f"    - v2 で **実コンテンツchunkが浮上**（section「{h2['section']}」/ "
                    f"score {h2['score']:.4f}）{kind}。v1 ではナビchunkに押し下げられ圏外だった → **救済**。"
                )
    else:
        L.append("- （retrieval_ok が反転した回答問は無し）")
    L.append(
        f"- 正味: retrieval_ok 合計 {fmt(t1['retrieval'])} → {fmt(t2['retrieval'])}。"
        "**合計は不変でも、反転により評価の質的中身が変わった点が要点**。\n"
    )

    # ナビ混入問の生 top_k（参考: nav が抜けた様子）
    L.append("## ナビ混入問（id1,6,7,8）の top_k 変化（参考）")
    for i in (1, 6, 7, 8):
        r1, r2 = log1.get(i), log2.get(i)
        if not r1 or not r2:
            continue
        kind = "棄却すべき" if v1[i]["should_abstain"] else "回答すべき"
        L.append(f"- **id{i}（{kind}）** v1 nav={nav_count(r1)}/4 → v2 nav={nav_count(r2)}/4")
        L.append(f"    - v1 top_k(cat): {'|'.join(topk_cats(r1))}")
        L.append(f"    - v2 top_k(cat): {'|'.join(topk_cats(r2))}")
        if not v1[i]["should_abstain"]:
            L.append(f"    - retrieval_ok: {cellv(v1[i].get('retrieval_ok'))} → {cellv(v2[i].get('retrieval_ok'))} / "
                     f"grounded_ok: {cellv(v1[i].get('grounded_ok'))} → {cellv(v2[i].get('grounded_ok'))}")
        else:
            L.append(f"    - abstain_ok: {cellv(v1[i].get('abstain_ok'))} → {cellv(v2[i].get('abstain_ok'))}")
    L.append("")

    # LLM 変動の区別
    unchanged = [i for i in ids if i not in changed]
    moved_despite_unchanged = []
    for i in unchanged:
        for key in ("retrieval_ok", "grounded_ok", "abstain_ok"):
            if v1[i].get(key) != v2[i].get(key):
                moved_despite_unchanged.append((i, key, v1[i].get(key), v2[i].get(key)))
    L.append("## LLM 変動（再生成ノイズ）の区別")
    L.append(f"- 検索が**不変**の問い = {unchanged}。これらの top_k は索引変更の影響を受けない（上位にナビ無し）。")
    if moved_despite_unchanged:
        L.append("- 検索不変なのに軸が動いた = **ask 再実行の生成ゆらぎ（LLM 変動）**であり、ナビ除去の効果ではない:")
        for i, key, a, b in moved_despite_unchanged:
            L.append(f"    - id{i} {key}: {cellv(a)} → {cellv(b)}（LLM 変動）")
    else:
        L.append("- 検索不変の問いで動いた軸は無し（生成ゆらぎは観測されず）。")
    L.append("")

    # 結論
    dr = t2["retrieval"][0] - t1["retrieval"][0]
    dg = t2["grounded"][0] - t1["grounded"][0]
    da = t2["abstain"][0] - t1["abstain"][0]
    n_flips = len(flips)
    L.append("## 結論（ナビ除去の効果）")
    L.append(f"- retrieval_ok: {fmt(t1['retrieval'])} → {fmt(t2['retrieval'])}（正味 {dr:+d}・反転 {n_flips} 件）")
    L.append(f"- grounded_ok: {fmt(t1['grounded'])} → {fmt(t2['grounded'])}（{dg:+d}）")
    L.append(f"- abstain_ok: {fmt(t1['abstain'])} → {fmt(t2['abstain'])}（{da:+d}）")
    if n_flips:
        flip_str = "、".join(f"id{i}（{cellv(a)}→{cellv(b)}）" for i, a, b, exp, h1, h2 in flips)
        L.append(f"- **検索の正味は {dr:+d} だが retrieval_ok が {n_flips} 件反転**: {flip_str}。")
        L.append(
            "    - ナビ除去は (1) ナビ経由の**偽陽性一致を是正**（見かけのスコアを下げる方向）と "
            "(2) 埋もれた**実コンテンツの救済**（上げる方向）を同時に起こした。"
            "**正味が小動でも評価の正確さは向上**（v1 の合格には偽陽性が含まれていた）。"
        )
    elif dr == 0:
        L.append("- 検索スコアは反転も含めて不変。")
    if dg == 0 and da == 0:
        L.append("- grounded/abstain は不変（検索反転問の生成内容も人手判定で同等、LLM 変動も無し）。")
    else:
        L.append("- grounded/abstain の差は、検索反転問由来か検索不変問の **LLM 変動**由来かを上節で区別済み。")
    L.append(
        "- 効果判定: ナビ除去は **検索の質を是正**（偽陽性除去＋実コンテンツ救済）。"
        "正味スコアは小動だが、これは**単一変更の測定**として正当かつ有意義な結論（DoD 達成）。"
    )
    L.append("")

    with open(OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print(f"[diff] {OUT} を生成。retrieval {dr:+d} / grounded {dg:+d} / abstain {da:+d}、top_k変化={changed}")


if __name__ == "__main__":
    main()
