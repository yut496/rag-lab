"""定数・設定値（大田区ごみRAG）。値のみを持ち、ロジックは持たない。"""

import os
import re

# ---- コーパス: 7カテゴリに対応する固定ページ ----
BASE = "https://www.city.ota.tokyo.jp"
UA = "Mozilla/5.0 (compatible; ota-gomi-rag/0.1; personal research)"  # gov サイトの既定UA弾き対策

# 許可カテゴリ（委託者確定・増減禁止）。全チャンクの category はこの7値のいずれかに正規化される。
CATEGORIES = (
    "プラスチック",
    "可燃ごみ",
    "不燃ごみ",
    "粗大ごみ",
    "資源",
    "区で収集できないもの",
    "集積所",
)

# source_url -> 7カテゴリ の明示マッピング。category はこの辞書だけで決定する（見出しは使わない）。
URL_CATEGORY = {
    BASE + "/seikatsu/gomi/shigentogomi/purasuhcikunodashikata.html": "プラスチック",
    BASE + "/seikatsu/gomi/shigentogomi/kanen.html": "可燃ごみ",
    BASE + "/seikatsu/gomi/shigentogomi/funen.html": "不燃ごみ",
    BASE + "/seikatsu/gomi/shigentogomi/sodai.html": "粗大ごみ",
    BASE + "/seikatsu/gomi/shigentogomi/shigen.html": "資源",
    BASE + "/seikatsu/gomi/shigentogomi/syuusyuu.html": "区で収集できないもの",
    BASE + "/seikatsu/gomi/shigentogomi/manners.html": "集積所",
}
# 取得対象は7カテゴリ対応ページのみ（旧41ページ -> 7ページ。非該当ページは除外）。
SOURCES = sorted(URL_CATEGORY)

# ---- 埋め込み / 索引 / 生成 ----
EMB_MODEL = "intfloat/multilingual-e5-base"  # JA特化なら cl-nagoya/ruri-base に差替 (prefix方式が異なる点に注意)
INDEX_PATH = "index.npz"
SCORE_FLOOR = 0.78  # これ未満なら棄却 (e5正規化cos。要校正)
TOP_K = 4
LLM_MODEL = os.environ.get(
    "OTA_LLM_MODEL", "claude-haiku-4-5"
)  # 生成モデル(安価なPoC既定)

# ---- 境界ルーター用パターン / 棄却メッセージ ----
ITEM_CLASSIFY_PAT = re.compile(
    r"(は何ごみ|は何ゴミ|何ごみ\?|何ゴミ\?|どのごみ|どう捨て|捨て方は\s*$)"
)
COLLECTION_DAY_PAT = re.compile(r"(収集日|何曜日|いつ出せ|いつ収集|回収日)")
OUT_OF_SCOPE_MSG = {
    "collection_day": "収集日は集積所の看板または大田区ごみ分別アプリでご確認ください（住所により異なります）。",
    "item_classify": "品目ごとの分別区分は大田区ごみ分別アプリの分別辞典でご確認ください。本回答は出し方ルールの解説に限定しています。",
}

# ---- チャンク化 / 観測 ----
MAX_CHARS = 700  # セクションが長い場合の分割上限
TEXT_HEAD = 200  # observe の top_k に載せる本文の先頭文字数
ID14_CATEGORY = "区で収集できないもの"  # observe の id14（モバイルバッテリー）特記対象
