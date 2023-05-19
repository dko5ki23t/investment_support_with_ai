# データ構造

## 株価情報データ

### 生成元Pythonコード

_investment\_support\_with\_ai_/source/fetch_data/fetch_data.py

### 格納場所

_investment\_support\_with\_ai_/data/fetched_data/

### ファイル名

_銘柄コード_.pkl

### 形式

DataFrame(pickle化)

### 構造

| カラム名 | 型 | 説明・補足 |
| ---- | ---- | ---- |
| code | numpy.int64 | 銘柄コード |
| name | str | 銘柄名 |
| timestamp | pandas._libs.tslibs.timestamps.Timestamp | 対象時刻のタイムスタンプ |
| date | datetime.date | 対象の日時(日時のみの比較に使うとき便利なため) |
| open | numpy.float64 | 始値 |
| high | numpy.float64 | 高値 |
| low | numpy.float64 | 安値 |
| close | numpy.float64 | 終値 |
| volume | numpy.float64 | 出来高 |
| day | numpy.int64 | 情報取得時からの通算日数。グラフ作成等に使用。TODO:DataFrameのindexと同じになるので不要？ |
| real/model | str | モデル名。YahooのAPIで取得した実測データの場合は'real'、機械学習等のモデルから得た予測データの場合はそのモデル名が格納される。グラフ作成等に使用。 |




## 機械学習モデルデータ

### 生成元Pythonコード

_investment\_support\_with\_ai_/source/learn/learn.py

### 格納場所

_investment\_support\_with\_ai_/data/learning_data/

### ファイル名

_銘柄コード_.pkl

### 形式

TODO(pickle化)

### 構造

TODO




## 株価予実データ

### 生成元Pythonコード

* _investment\_support\_with\_ai_/source/estimate/estimate.py

  - モデルによる予測データを格納

* _investment\_support\_with\_ai_/source/analyze/analyze.py

  - 実データを追記

### 格納場所

_investment\_support\_with\_ai_/data/estimate_data/

### ファイル名

analyze _x:x日後の予測_.pkl

### 形式

DataFrame(pickle化)

### 構造

| カラム名 | 型 | 説明・補足 |
| ---- | ---- | ---- |
| code | numpy.int64 | 銘柄コード |
| name | str | 銘柄名 |
| timestamp | pandas._libs.tslibs.timestamps.Timestamp | 対象時刻のタイムスタンプ |
| date | datetime.date | 対象の日時(日時のみの比較に使うとき便利なため) |
| model | str | 予測モデル名 |
| price | numpy.float64 | 予測時の値段(per 1株) |
| term | numpy.int64 | 予測期間(x日後の終値を予測) |
| predict | numpy.float32 | 予測値(per 1株) |
| predict gain | numpy.float64 | 予測損益(per 1株)。 _predict gain_ = _(predict - price)_ |
| actual | numpy.float64 | 実測値(per 1株) |
| msr | numpy.float64 | MSR |
| delta value | numpy.float64 | 予実差(値) |
| delta ration | numpy.float64 | 予実差(割合)。 _delta ration_ = _(actual / predict)_ |
