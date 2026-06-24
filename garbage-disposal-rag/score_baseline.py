"""Phase 5-A: observation_log.jsonl を新しい3軸ルールで再採点し、ベースラインを出力する。

これは「採点ルールの変更」であり、ask パイプラインの再実行ではない。
system（route / SCORE_FLOOR / プロンプト / index）は一切読まず・変えず、観測ログを採点するだけ。

  .venv/bin/python score_baseline.py [log.jsonl] [out.csv] [out.md]
    引数省略時は v1（observation_log.jsonl / scoring_baseline.csv / scoring_summary.md）。
    Phase 5-B は: score_baseline.py observation_log_v2.jsonl scoring_v2.csv scoring_v2_summary.md
    1) CSV が無ければ: ログを採点して生成（grounded/abstain は REVIEW 付き）
    2) CSV を「真実の源」として集計し summary.md を生成
       （= REVIEW を人手確定した後に再実行すると、確定値で summary が更新される）

  ※ CSV が既に在る場合は再生成しない（人手確定を保護）。再ベースライン時は CSV を削除して再実行。
  ※ 採点ルールは Phase 5-A 確定版を不変のまま使う（引数で変えるのはパスのみ）。

採点方針:
- retrieval_ok だけが完全に機械判定可能（expected_category が top_k のカテゴリ集合に入るか）。
- grounded_ok / abstain_ok は LLM 応答の文面判断が要る。生成回答は「ごみ減量推進課」等の連絡先や
  「確認できません」を、実質回答でもフッター/部分保険として含むため、キーワードで降参/実質回答を
  機械分離できない。よって自動 True は「system が実際に棄却した(abstained=True)」場合のみとし、
  それ以外は REVIEW（人手判定待ち）にする。自動で False は出さない。
"""

import csv
import json
import os
import sys

LOG_PATH = "observation_log.jsonl"
GOLDENSET_PATH = "goldenset.json"
CSV_PATH = "scoring_baseline.csv"
SUMMARY_PATH = "scoring_summary.md"

FIELDS = [
    "id", "should_abstain", "expected_category", "top_k_categories",
    "retrieval_ok", "grounded_ok", "abstain_ok", "note",
]


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_abstain_reasons(path):
    """goldenset から id -> abstain_reason を取る（note 補強用・任意）。無ければ空 dict。"""
    try:
        with open(path, encoding="utf-8") as f:
            gold = json.load(f)
    except FileNotFoundError:
        return {}
    return {g["id"]: g.get("abstain_reason") for g in gold}


def score_record(r, reasons):
    """1レコードを3軸で初期採点して行 dict を返す（grounded/abstain は REVIEW 付き）。"""
    should_abstain = r["should_abstain"]
    cats = [h["category"] for h in r["top_k"]]
    abstained = r["abstained"]

    if not should_abstain:
        retrieval_ok = r["expected_category"] in cats
        if abstained:
            grounded_ok = True  # 棄却＝何も断定していない
            note = "system棄却→grounded自動True（断定なし）"
        else:
            grounded_ok = "REVIEW"
            note = "実質回答→参照チャンクとの整合は人手判定(REVIEW)"
        abstain_ok = "N/A"
    else:
        retrieval_ok = "N/A"
        grounded_ok = "N/A"
        if abstained:
            abstain_ok = True
            note = f"system棄却(routed={r['routed']})"
        else:
            abstain_ok = "REVIEW"
            note = "system応答(abstained=False)→降参か誤断定かは人手判定(REVIEW)"
        reason = reasons.get(r["id"])
        if reason:
            note = f"[{reason}] {note}"

    return {
        "id": r["id"],
        "should_abstain": should_abstain,
        "expected_category": r["expected_category"] if r["expected_category"] else "—",
        "top_k_categories": "|".join(cats) if cats else "—",
        "retrieval_ok": retrieval_ok,
        "grounded_ok": grounded_ok,
        "abstain_ok": abstain_ok,
        "note": note,
    }


def old_scores(records):
    """旧採点を同じログから再導出（差分の自己整合のため）。
    旧 retrieval = top-1 category 一致 / 旧 abstain = system が棄却(abstained=True) のみ。
    """
    retr = retr_total = abst = abst_total = 0
    for r in records:
        if r["should_abstain"]:
            abst_total += 1
            abst += int(bool(r["abstained"]))
        else:
            retr_total += 1
            cats = [h["category"] for h in r["top_k"]]
            if cats and cats[0] == r["expected_category"]:
                retr += 1
    return (retr, retr_total), (abst, abst_total)


def write_csv(rows, path):
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)


def _cell(v):
    if v == "True":
        return True
    if v == "False":
        return False
    return v  # "N/A" / "REVIEW"


def read_rows(path):
    """scoring_baseline.csv（人手確定を含む）を読み、型を戻して返す。"""
    with open(path, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["id"] = int(r["id"])
        for k in ("should_abstain", "retrieval_ok", "grounded_ok", "abstain_ok"):
            r[k] = _cell(r[k])
    return rows


def build_summary(rows, old):
    answer = [x for x in rows if x["should_abstain"] is False]
    abstain = [x for x in rows if x["should_abstain"] is True]

    def tally(items, key):
        ok = sum(1 for x in items if x[key] is True)
        review = sum(1 for x in items if x[key] == "REVIEW")
        return ok, review

    na, nb = len(answer), len(abstain)
    r_ok, r_rev = tally(answer, "retrieval_ok")
    g_ok, g_rev = tally(answer, "grounded_ok")
    a_ok, a_rev = tally(abstain, "abstain_ok")
    (old_retr, old_retr_n), (old_abst, old_abst_n) = old
    resolved = (g_rev == 0 and a_rev == 0)

    def rev(n):
        return "（REVIEW 確定済み）" if n == 0 else f"（REVIEW {n} 件＝人手判定待ち）"

    L = []
    state = "REVIEW 人手確定・最終版" if resolved else "REVIEW 人手判定待ち"
    L.append(f"# Phase 5-A 再採点ベースライン（{state}）\n")
    L.append("採点ルールの変更のみ。ask は再実行せず、system（route/SCORE_FLOOR/プロンプト/index）は無変更。")
    L.append("スコアは scoring_baseline.csv を真実の源として集計（REVIEW の人手確定を反映）。\n")

    L.append("## 3軸スコア")
    L.append(f"- **retrieval_ok**: {r_ok}/{na}（回答問・機械判定）")
    L.append(f"- **grounded_ok**: {g_ok}/{na}{rev(g_rev)}")
    L.append(f"- **abstain_ok**: {a_ok}/{nb}{rev(a_rev)}\n")

    L.append("## 旧採点（同じログから再導出・自己整合チェック）")
    L.append(f"- top-1 category 一致: {old_retr}/{old_retr_n}（Phase 4 の 4/8 を再現）")
    L.append(f"- 棄却成功(system棄却のみ): {old_abst}/{old_abst_n}（Phase 4 の 2/7 を再現）\n")

    L.append("## 差分（定義変更で数字がどう動いたか）")
    L.append(
        f"- **検索**: top-1一致 {old_retr}/{old_retr_n} → top-K集合 {r_ok}/{na}"
        f"（{r_ok - old_retr:+d}）。top-1 で外れるが top-4 に expected が入る問いを救済（機械）。"
    )
    if resolved:
        L.append(
            f"- **棄却**: system棄却 {old_abst}/{old_abst_n} → {a_ok}/{nb}（{a_ok - old_abst:+d}）。"
            f"内訳: system棄却 {old_abst} ＋ 回答本文の降参を人手確定 {a_ok - old_abst}。"
        )
        L.append(f"- **grounded**: 新設軸 → {g_ok}/{na}（人手確定）。\n")
    else:
        L.append(
            f"- **棄却**: system棄却 {old_abst}/{old_abst_n} → 自動True {a_ok}/{nb} ＋ REVIEW {a_rev} 件"
            "（新定義の回答降参分は人手確定待ち）。"
        )
        L.append(f"- **grounded**: 新設軸。実質回答の参照整合は機械判定不可のため REVIEW {g_rev} 件。\n")

    if resolved:
        L.append("## 残る失敗（Phase 5-B の改善対象）")
        fail_lines = []
        for x in sorted(rows, key=lambda r: r["id"]):
            if x["retrieval_ok"] is False:
                fail_lines.append(
                    f"- id{x['id']} retrieval_ok=False: expected「{x['expected_category']}」が "
                    f"top_k({x['top_k_categories']}) に不在"
                )
            if x["grounded_ok"] is False:
                fail_lines.append(f"- id{x['id']} grounded_ok=False: 回答が参照チャンクと不整合")
            if x["abstain_ok"] is False:
                fail_lines.append(f"- id{x['id']} abstain_ok=False: 棄却すべきだが実質回答（棄却漏れ）")
        L.extend(fail_lines if fail_lines else ["（False なし）"])
        L.append("")
    else:
        L.append("## REVIEW 一覧（人手判定待ち・本文は observation_log.jsonl の generated_answer を参照）")
        L.append("| id | 軸 | 理由 |")
        L.append("|---|---|---|")
        for x in rows:
            if x["grounded_ok"] == "REVIEW":
                L.append(f"| {x['id']} | grounded_ok | {x['note']} |")
            if x["abstain_ok"] == "REVIEW":
                L.append(f"| {x['id']} | abstain_ok | {x['note']} |")
        L.append("")

    L.append("## 注記")
    L.append("- 自動で False を出すのは retrieval_ok のみ。grounded/abstain は「自動 True か REVIEW」で、REVIEW は scoring_baseline.csv 上で人手確定する。")
    L.append("- CSV が既存なら本スクリプトは CSV を再生成せず（人手確定を保護）、CSV を集計して summary を更新する。再ベースライン時は CSV を削除して再実行。")
    return "\n".join(L)


def main(log_path=LOG_PATH, csv_path=CSV_PATH, summary_path=SUMMARY_PATH):
    records = load_jsonl(log_path)
    reasons = load_abstain_reasons(GOLDENSET_PATH)
    if not os.path.exists(csv_path):
        rows = sorted((score_record(r, reasons) for r in records), key=lambda x: x["id"])
        write_csv(rows, csv_path)
        print(f"[score] 初回採点 -> {csv_path}（REVIEW を人手確定後に再実行で summary 確定）")
    rows = read_rows(csv_path)
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(build_summary(rows, old_scores(records)) + "\n")
    print(f"[score] summary 生成 -> {summary_path}（CSV を真実の源として集計）")


if __name__ == "__main__":
    # 引数: [log.jsonl] [out.csv] [out.md]（省略時は v1 既定）
    args = sys.argv[1:]
    main(*args) if args else main()
