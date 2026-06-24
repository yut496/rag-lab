# Phase 5-A 再採点ベースライン（REVIEW 人手確定・最終版）

採点ルールの変更のみ。ask は再実行せず、system（route/SCORE_FLOOR/プロンプト/index）は無変更。
スコアは scoring_baseline.csv を真実の源として集計（REVIEW の人手確定を反映）。

## 3軸スコア
- **retrieval_ok**: 7/8（回答問・機械判定）
- **grounded_ok**: 8/8（REVIEW 確定済み）
- **abstain_ok**: 6/7（REVIEW 確定済み）

## 旧採点（同じログから再導出・自己整合チェック）
- top-1 category 一致: 4/8（Phase 4 の 4/8 を再現）
- 棄却成功(system棄却のみ): 2/7（Phase 4 の 2/7 を再現）

## 差分（定義変更で数字がどう動いたか）
- **検索**: top-1一致 4/8 → top-K集合 7/8（+3）。top-1 で外れるが top-4 に expected が入る問いを救済（機械）。
- **棄却**: system棄却 2/7 → 6/7（+4）。内訳: system棄却 2 ＋ 回答本文の降参を人手確定 4。
- **grounded**: 新設軸 → 8/8（人手確定）。

## 残る失敗（Phase 5-B の改善対象）
- id1 abstain_ok=False: 棄却すべきだが実質回答（棄却漏れ）
- id14 retrieval_ok=False: expected「区で収集できないもの」が top_k(不燃ごみ|プラスチック|プラスチック|集積所) に不在

## 注記
- 自動で False を出すのは retrieval_ok のみ。grounded/abstain は「自動 True か REVIEW」で、REVIEW は scoring_baseline.csv 上で人手確定する。
- CSV が既存なら本スクリプトは CSV を再生成せず（人手確定を保護）、CSV を集計して summary を更新する。再ベースライン時は CSV を削除して再実行。
