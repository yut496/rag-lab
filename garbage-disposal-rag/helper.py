"""チャンク化ヘルパ（見出しを保つ本文分割・<table> の行保持）。rag.py の chunk() が使う。"""

import json
import random
import re
import time

from constant import ID14_CATEGORY, MAX_CHARS


def split_keep_header(header, body):
    """セクションが長ければ分割。各断片の先頭に見出しを必ず付与。"""
    out, cur = [], ""
    for sent in re.split(r"(?<=。)", body):
        if len(cur) + len(sent) > MAX_CHARS and cur:
            out.append(f"【{header}】{cur.strip()}")
            cur = ""
        cur += sent
    if cur.strip():
        out.append(f"【{header}】{cur.strip()}")
    return out


def table_to_rows(table):
    """<table> を「1行=セルを ｜ で連結した文字列」のリストにする（行は分断しない単位）。"""
    rows = []
    for tr in table.find_all("tr"):
        cells = [c.get_text(" ", strip=True) for c in tr.find_all(["th", "td"])]
        cells = [c for c in cells if c]
        if cells:
            rows.append(" ｜ ".join(cells))
    return rows


def split_table_keep_header(header, rows):
    """表を見出し付きチャンクにまとめる。MAX_CHARS 超過時は行境界で分割し、行は決して割らない。"""
    out, cur = [], []
    for row in rows:
        if cur and sum(len(r) for r in cur) + len(row) > MAX_CHARS:
            out.append(f"【{header}】\n" + "\n".join(cur))
            cur = []
        cur.append(row)
    if cur:
        out.append(f"【{header}】\n" + "\n".join(cur))
    return out


def dump_for_inspection(chunks, categories, n_sources):
    """観察用の出力: 取得日 / category集計 / 全チャンクのダンプ / 無作為20件サンプル(v2)。"""
    today = time.strftime("%Y-%m-%d")
    # category 集計（全チャンクが許可7カテゴリに収まっているかを確認・表示する）
    counts = {}
    for c in chunks:
        counts[c.category] = counts.get(c.category, 0) + 1
    table = [f"  {cat}: {counts.get(cat, 0)}" for cat in categories]
    extra = {cat: n for cat, n in counts.items() if cat not in categories}
    print("[ingest] category 集計（許可7カテゴリ）:")
    for line in table:
        print(line)
    if extra:
        print(f"[warn] 7カテゴリ外の category が混入: {extra}")
    else:
        print("[ingest] OK: 全チャンクが7カテゴリ内")

    with open("corpus_date.txt", "w", encoding="utf-8") as f:
        f.write(f"corpus snapshot date: {today}\n")
        f.write(f"sources: {n_sources} pages\n")
        f.write(f"chunks: {len(chunks)}\n")
        f.write("category counts (whitelist of 7):\n")
        for line in table:
            f.write(line + "\n")
        if extra:
            f.write(f"[warn] outside-whitelist categories: {extra}\n")
    with open("chunks_dump.txt", "w", encoding="utf-8") as f:
        for i, c in enumerate(chunks):
            f.write(
                f"=== [{i}] category={c.category} | section={c.section} | {c.source_url}\n"
            )
            f.write(c.text + "\n\n")
    rng = random.Random(42)  # seed固定で再現可能
    sample = rng.sample(chunks, min(20, len(chunks)))
    with open("chunks_sample_v2.txt", "w", encoding="utf-8") as f:
        f.write(
            f"# 無作為20件サンプル v2（seed=42 / 全{len(chunks)}件中 / category正規化後）\n\n"
        )
        for c in sample:
            f.write(f"--- category={c.category} | section={c.section}\n{c.text}\n\n")
    print(
        "[ingest] inspection dump -> corpus_date.txt / chunks_dump.txt / chunks_sample_v2.txt"
    )


def write_observation_log(records, path="observation_log.jsonl"):
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def write_observation_summary(
    records, score_floor, top_k, path="observation_summary.md"
):
    L = []
    L.append("# Phase 4 観測サマリ（ask パイプライン一周）\n")
    L.append(
        f"- 対象: goldenset {len(records)}問 / SCORE_FLOOR={score_floor}（固定・未調整） / TOP_K={top_k}"
    )
    L.append(
        "- 本フェーズは観測のみ。閾値・プロンプト・route・index は未変更。失敗は失敗のまま記録。\n"
    )

    # 1) route 分類一覧
    L.append("## 1. route 分類")
    L.append("| id | routed | query |")
    L.append("|---|---|---|")
    for r in records:
        L.append(f"| {r['id']} | {r['routed']} | {r['query']} |")
    L.append("")

    # 2) 棄却すべき問い（should_abstain=true）を実際に棄却できたか
    L.append("## 2. 棄却の成否（should_abstain=true の問い）")
    L.append("| id | 実際にabstain | 判定 | reason | query |")
    L.append("|---|---|---|---|---|")
    abst_ok = abst_total = 0
    for r in records:
        if not r["should_abstain"]:
            continue
        abst_total += 1
        ok = bool(r["abstained"])
        abst_ok += ok
        reason = (
            r["routed"]
            if r["routed"] != "rule_explanation"
            else f"top={r['top_score']}(<floor?)"
        )
        L.append(
            f"| {r['id']} | {r['abstained']} | {'○' if ok else '×'} | {reason} | {r['query']} |"
        )
    L.append(f"\n→ 棄却成功 {abst_ok}/{abst_total}\n")

    # 3) 回答問（should_abstain=false）の top_category 一致
    L.append("## 3. 回答問の category 一致（should_abstain=false）")
    L.append("| id | expected | top_category | 一致 | abstained | top_score | query |")
    L.append("|---|---|---|---|---|---|---|")
    cat_ok = cat_total = 0
    for r in records:
        if r["should_abstain"]:
            continue
        cat_total += 1
        top_cat = r["top_k"][0]["category"] if r["top_k"] else None
        match = (not r["abstained"]) and (top_cat == r["expected_category"])
        cat_ok += match
        shown_cat = "(棄却された)" if r["abstained"] else top_cat
        L.append(
            f"| {r['id']} | {r['expected_category']} | {shown_cat} | {'○' if match else '×'} | {r['abstained']} | {r['top_score']} | {r['query']} |"
        )
    L.append(f"\n→ top_category 一致 {cat_ok}/{cat_total}\n")

    # 4) id14 特記
    L.append("## 4. 特記: id14（モバイルバッテリー）の top_k")
    id14 = next((r for r in records if r["id"] == 14), None)
    if id14 is None:
        L.append("（id14 が見つからない）\n")
    else:
        cats = [h["category"] for h in id14["top_k"]]
        has = ID14_CATEGORY in cats
        L.append(f"- query: {id14['query']}")
        L.append(
            f"- expected_category: {id14['expected_category']} / abstained: {id14['abstained']} / top_score: {id14['top_score']}"
        )
        L.append(
            f"- **top_k に「{ID14_CATEGORY}」カテゴリが入ったか: {'YES' if has else 'NO'}**"
        )
        L.append(f"- top_k categories（順）: {cats}\n")

    # 5) 観測された不一致（改善候補・本フェーズでは直さない）
    L.append("## 5. 観測された不一致（改善候補・本フェーズでは直さない）")
    issues = []
    for r in records:
        if r["should_abstain"] and not r["abstained"]:
            issues.append(
                f"- id{r['id']} 棄却漏れ: 棄却すべきだが回答（routed={r['routed']}, top={r['top_score']}）— {r['query']}"
            )
        elif (not r["should_abstain"]) and r["abstained"]:
            issues.append(
                f"- id{r['id']} 過剰棄却: 回答すべきだが棄却（top={r['top_score']} < SCORE_FLOOR={score_floor}）— {r['query']}"
            )
        elif (not r["should_abstain"]) and (not r["abstained"]):
            top_cat = r["top_k"][0]["category"] if r["top_k"] else None
            if top_cat != r["expected_category"]:
                issues.append(
                    f"- id{r['id']} category不一致: top={top_cat} 期待={r['expected_category']}（top_score={r['top_score']}）— {r['query']}"
                )
    L.extend(issues if issues else ["（機械検出の不一致なし）"])
    L.append("")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(L))
