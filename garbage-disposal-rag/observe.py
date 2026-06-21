"""Phase 4: ask パイプライン（route→retrieve→棄却→生成）を goldenset 全15問で一周し「観測」する。

目的は観測であって精度向上ではない。**SCORE_FLOOR / プロンプト / route / index を一切変更しない**。
rag.py の関数・定数をそのまま使い、各問を `rag.ask()` で実走させる。失敗は失敗のまま記録する。

実行（ネットワーク必須＝call_llm が Anthropic を呼ぶ。APIキーは rag-lab/.env から load_dotenv で読む・export 不要）:
  cd garbage-disposal-rag && uv run python observe.py [goldenset.json]
出力:
  observation_log.jsonl … 1問=1行（top_k本文先頭を含む。再配布回避で gitignore）
  observation_summary.md … route分類・棄却成否・category一致・id14特記（本文ダンプ無し）
"""

import sys
import json

from dotenv import load_dotenv

import rag

load_dotenv()   # rag-lab 直下の .env から ANTHROPIC_API_KEY を os.environ に注入（test.py と同じ。export 不要・rag.py は無変更）。

TEXT_HEAD = 200                        # top_k に載せる本文の先頭文字数
ID14_CATEGORY = "区で収集できないもの"   # 特記対象（モバイルバッテリーの期待カテゴリ）


def observe_one(item):
    """1問を rag.ask() に通し、観測レコード(dict)を返す。ask/route は変更せずそのまま使う。"""
    q = item["query"]
    res = rag.ask(q)                   # ← 実パイプライン（route→retrieve→棄却→生成）
    routed = rag.route(q)              # 観測用に route 判定を別取得（ask 内部と同じ判定）
    hits = res.get("hits", [])
    top_k = [
        {
            "score": round(float(score), 4),
            "category": m.get("category"),
            "section": m.get("section"),
            "text_head": (m.get("text") or "")[:TEXT_HEAD],
        }
        for score, m in hits
    ]
    top_score = res.get("top_score")
    if top_score is None and top_k:
        top_score = top_k[0]["score"]
    return {
        "id": item["id"],
        "query": q,
        "should_abstain": item["should_abstain"],
        "expected_category": item["expected_category"],
        "routed": routed,
        "top_score": round(float(top_score), 4) if top_score is not None else None,
        "top_k": top_k,
        "generated_answer": res.get("answer"),
        "abstained": res.get("abstain"),
    }


def write_log(records, path="observation_log.jsonl"):
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def write_summary(records, path="observation_summary.md"):
    L = []
    L.append("# Phase 4 観測サマリ（ask パイプライン一周）\n")
    L.append(f"- 対象: goldenset {len(records)}問 / SCORE_FLOOR={rag.SCORE_FLOOR}（固定・未調整） / TOP_K={rag.TOP_K}")
    L.append("- 本フェーズは観測のみ。閾値・プロンプト・route・index は未変更。失敗は失敗のまま記録。\n")

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
        reason = r["routed"] if r["routed"] != "rule_explanation" else f"top={r['top_score']}(<floor?)"
        L.append(f"| {r['id']} | {r['abstained']} | {'○' if ok else '×'} | {reason} | {r['query']} |")
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
        L.append(f"| {r['id']} | {r['expected_category']} | {shown_cat} | {'○' if match else '×'} | {r['abstained']} | {r['top_score']} | {r['query']} |")
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
        L.append(f"- expected_category: {id14['expected_category']} / abstained: {id14['abstained']} / top_score: {id14['top_score']}")
        L.append(f"- **top_k に「{ID14_CATEGORY}」カテゴリが入ったか: {'YES' if has else 'NO'}**")
        L.append(f"- top_k categories（順）: {cats}\n")

    # 5) 観測された不一致（改善候補・本フェーズでは直さない）
    L.append("## 5. 観測された不一致（改善候補・本フェーズでは直さない）")
    issues = []
    for r in records:
        if r["should_abstain"] and not r["abstained"]:
            issues.append(f"- id{r['id']} 棄却漏れ: 棄却すべきだが回答（routed={r['routed']}, top={r['top_score']}）— {r['query']}")
        elif (not r["should_abstain"]) and r["abstained"]:
            issues.append(f"- id{r['id']} 過剰棄却: 回答すべきだが棄却（top={r['top_score']} < SCORE_FLOOR={rag.SCORE_FLOOR}）— {r['query']}")
        elif (not r["should_abstain"]) and (not r["abstained"]):
            top_cat = r["top_k"][0]["category"] if r["top_k"] else None
            if top_cat != r["expected_category"]:
                issues.append(f"- id{r['id']} category不一致: top={top_cat} 期待={r['expected_category']}（top_score={r['top_score']}）— {r['query']}")
    L.extend(issues if issues else ["（機械検出の不一致なし）"])
    L.append("")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(L))


def main():
    gpath = sys.argv[1] if len(sys.argv) > 1 else "goldenset.json"
    goldenset = json.load(open(gpath, encoding="utf-8"))
    records = []
    for item in goldenset:
        print(f"[observe] id={item['id']}: {item['query'][:28]} ...")
        records.append(observe_one(item))
    write_log(records)
    write_summary(records)
    print(f"[observe] {len(records)} 問を観測 -> observation_log.jsonl / observation_summary.md")


if __name__ == "__main__":
    main()
