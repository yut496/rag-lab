"""ゴールデンセット gold.json のスキーマ検証スクリプト（Phase 3 / Phase 2 Hotfix 後の実スキーマ準拠）。

質問内容は委託者（人間）が作成する。本スクリプトは内容を生成・改変せず、
**構造と内訳要件のみ** を機械的に検証して PASS/FAIL を一覧表示する。

データ形式: 手編集しやすい整形 JSON 配列（1要素=1問）。
  例（gold_template.json も同形式）:
  [
    {
      "id": 1,
      "query": "",
      "source": "synthetic",
      "expected_category": "可燃ごみ",
      "should_abstain": false,
      "abstain_reason": null,
      "r74_changed": false,
      "smoke": false,
      "ideal_answer": "",
      "note": ""
    }
  ]

実行:
  python validate_gold.py [gold.json]   # 省略時は ./gold.json
  （標準ライブラリのみ。uv run でなくても動く。）

========================================================================
フィールド定義（1要素 = 1問）
========================================================================
  id                : int        必須・一意・正の整数（連番で採番）。
  query             : str        必須。問い本文（人間が記入）。内容は検証しない。
  source            : str        必須。出どころタグ。外部出典系（"official_faq" / "vendor_faq" 等）/
                                  外部出典なし（"synthetic"＝合成 / "self_authored"＝自作）。
  note              : str|null   出どころの補足。外部出典 source（synthetic / self_authored 以外）は必須（非空文字列）。
                                  synthetic / self_authored は空文字 / null でよい（source タグ自体が出どころのため）。
  should_abstain    : bool       必須。棄却すべき問いか。
  abstain_reason    : str|null   should_abstain==true のとき必須で次のいずれか:
                                  "item_classify"  … 分別判定（これは何ゴミ?）
                                  "collection_day" … 収集日（うちの収集日は?）
                                  "not_in_corpus"  … コーパスに無い品目
                                  "too_broad"      … 質問が漠然としすぎて答えられない
                                should_abstain==false のとき null。
  expected_category : str|null   should_abstain==false のとき非null＝下記7カテゴリのいずれか。
                                  should_abstain==true のとき null。
  r74_changed       : bool       必須。R7.4改定で正解が変わった内容か。
  smoke             : bool       必須。スモーク5問のメンバーか（全体で == 5）。
  ideal_answer      : str|null   任意。理想回答（文字列か null。一致は採点条件にしない）。

許可カテゴリ（rag.py の CATEGORIES と一致させること。Phase 2 Hotfix で確定）:
  プラスチック / 可燃ごみ / 不燃ごみ / 粗大ごみ / 資源 / 区で収集できないもの / 集積所

内訳要件（集計検証）:
  - 総数 == 15
  - id が一意
  - should_abstain==true の数 >= 4
  - abstain_reason ∈ {item_classify, collection_day} の数 >= 3
  - abstain_reason == not_in_corpus の数 >= 1
  - r74_changed==true の数 >= 1
  - smoke==true の数 == 5
  - 外部出典 source（synthetic / self_authored 以外）は note が空でない
  - should_abstain==false は expected_category が「7カテゴリのいずれか」
  - should_abstain==true は expected_category が null
========================================================================
"""

import sys
import json
import collections

REQUIRED_FIELDS = [
    "id", "query", "source", "note",
    "should_abstain", "abstain_reason", "expected_category",
    "r74_changed", "smoke",
]
ABSTAIN_REASONS = {"item_classify", "collection_day", "not_in_corpus", "too_broad"}
# note（出どころ）が不要な source。外部出典が無い＝自作・合成は空でよい。
NO_NOTE_SOURCES = {"synthetic", "self_authored"}
# rag.py の CATEGORIES と一致させる（Phase 2 Hotfix で確定した7カテゴリ）。
CATEGORIES = {
    "プラスチック", "可燃ごみ", "不燃ごみ", "粗大ごみ",
    "資源", "区で収集できないもの", "集積所",
}


def load_gold(path):
    """整形 JSON 配列を読み込む。トップレベルが配列でなければ ValueError。"""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("トップレベルは JSON 配列であるべき（[ {...}, {...} ]）")
    return data


def structural_issues(objs):
    """各要素のフィールド・型・整合性を検査し、問題メッセージのリストを返す。"""
    issues = []
    for i, o in enumerate(objs):
        if not isinstance(o, dict):
            issues.append(f"#{i}: オブジェクトでない（{type(o).__name__}）")
            continue
        tag = f"#{i} (id={o.get('id', '?')})"
        # 必須フィールド
        for fld in REQUIRED_FIELDS:
            if fld not in o:
                issues.append(f"{tag}: フィールド '{fld}' が無い")
        # id は正の整数（bool は除外）
        if "id" in o and (not isinstance(o["id"], int) or isinstance(o["id"], bool) or o["id"] <= 0):
            issues.append(f"{tag}: 'id' は正の整数であるべき（実際: {o.get('id')!r}）")
        # query / source は必須の文字列
        for fld in ("query", "source"):
            if fld in o and not isinstance(o[fld], str):
                issues.append(f"{tag}: '{fld}' は文字列であるべき")
        # note / ideal_answer は str か null（外部出典の note 非空は集計側で別途チェック）
        for fld in ("note", "ideal_answer"):
            if fld in o and o[fld] is not None and not isinstance(o[fld], str):
                issues.append(f"{tag}: '{fld}' は文字列か null であるべき")
        # 真偽値フィールド
        for fld in ("should_abstain", "r74_changed", "smoke"):
            if fld in o and not isinstance(o[fld], bool):
                issues.append(f"{tag}: '{fld}' は真偽値であるべき")
        # abstain_reason と should_abstain の整合
        sa = o.get("should_abstain")
        ar = o.get("abstain_reason")
        if sa is True:
            if ar not in ABSTAIN_REASONS:
                issues.append(f"{tag}: should_abstain==true なら abstain_reason は {sorted(ABSTAIN_REASONS)} のいずれか（実際: {ar!r}）")
        elif sa is False:
            if ar is not None:
                issues.append(f"{tag}: should_abstain==false なら abstain_reason は null（実際: {ar!r}）")
        # expected_category は str か null
        ec = o.get("expected_category")
        if ec is not None and not isinstance(ec, str):
            issues.append(f"{tag}: 'expected_category' は文字列か null")
    return issues


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "gold.json"
    try:
        objs = load_gold(path)
    except FileNotFoundError:
        print(f"✗ {path} が見つかりません。gold_template.json を雛形に作成してください。")
        sys.exit(2)
    except (json.JSONDecodeError, ValueError) as e:
        print(f"✗ {path} を JSON 配列として読めません: {e}")
        sys.exit(2)

    n = len(objs)
    checks = []   # (ok: bool, name: str, detail: str)

    def add(ok, name, detail=""):
        checks.append((ok, name, detail))

    # 1) 構造（フィールド・型・整合）
    issues = structural_issues(objs)
    add(not issues, "構造（必須フィールド・型・整合）",
        "\n      - " + "\n      - ".join(issues) if issues else "")

    # 2) 集計チェック
    add(n == 15, "総数 == 15", f"実際 {n}")

    valid = [o for o in objs if isinstance(o, dict)]
    ids = [o.get("id") for o in valid]
    dup = sorted(str(i) for i, c in collections.Counter(ids).items() if c > 1)
    add(not dup, "id が一意", f"重複 id: {dup}")

    ab = [o for o in valid if o.get("should_abstain") is True]
    add(len(ab) >= 4, "should_abstain==true の数 >= 4", f"実際 {len(ab)}")

    ar_ic = [o for o in valid if o.get("abstain_reason") in ("item_classify", "collection_day")]
    add(len(ar_ic) >= 3,
        "abstain_reason ∈ {item_classify, collection_day} の数 >= 3", f"実際 {len(ar_ic)}")

    ar_nc = [o for o in valid if o.get("abstain_reason") == "not_in_corpus"]
    add(len(ar_nc) >= 1, "abstain_reason == not_in_corpus の数 >= 1", f"実際 {len(ar_nc)}")

    r74 = [o for o in valid if o.get("r74_changed") is True]
    add(len(r74) >= 1, "r74_changed==true の数 >= 1", f"実際 {len(r74)}")

    sm = [o for o in valid if o.get("smoke") is True]
    add(len(sm) == 5, "smoke==true の数 == 5", f"実際 {len(sm)}")

    bad_note = [o.get("id") for o in valid
                if o.get("source") not in NO_NOTE_SOURCES and not str(o.get("note") or "").strip()]
    add(not bad_note, "外部出典 source（synthetic/self_authored 以外）は note 非空", f"note 空の id: {bad_note}")

    bad_ec_false = [o.get("id") for o in valid
                    if o.get("should_abstain") is False and not o.get("expected_category")]
    add(not bad_ec_false, "should_abstain==false は expected_category 非null", f"違反 id: {bad_ec_false}")

    bad_ec_cat = [o.get("id") for o in valid
                  if o.get("should_abstain") is False and o.get("expected_category")
                  and o.get("expected_category") not in CATEGORIES]
    add(not bad_ec_cat, "expected_category は7カテゴリのいずれか", f"カテゴリ外の id: {bad_ec_cat}")

    bad_ec_true = [o.get("id") for o in valid
                   if o.get("should_abstain") is True and o.get("expected_category") is not None]
    add(not bad_ec_true, "should_abstain==true は expected_category == null", f"違反 id: {bad_ec_true}")

    # 出力
    print(f"=== validate_gold: {path}  ({n} 件) ===")
    all_ok = True
    for ok, name, detail in checks:
        mark = "PASS" if ok else "FAIL"
        line = f"  [{mark}] {name}"
        if not ok and detail:
            line += f" — {detail}"
        print(line)
        all_ok = all_ok and ok
    print("=> " + ("ALL PASS ✅" if all_ok else "FAIL ❌（上記 FAIL を修正してください）"))
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
