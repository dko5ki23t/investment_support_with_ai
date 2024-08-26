# フォルダ構成

```
(リポジトリトップ)
 |- db
    |- estimates
    |  |- estimate_xxx           xxx(予想方法名)での予想結果が入るフォルダ
    |         yyyyy.json         yyyyy(銘柄コード)についての予想結果ファイル
    |- orders
    |  |- order_zz               zz(注文作成方法ごとに割り振られた番号)で作成された注文が入るフォルダ
    |         order_vvv_xxx.json vvv(市場コード)を対象としてxxx(予想方法名)での予想から作成された注文ファイル
    |- stock_data
    |  |- N225
    |         N225.parquet       日経平均の株価情報ファイル
    |  yyyyy.parquet             yyyyy(銘柄コード)の株価情報ファイル
    |- evaluate
              strategyzz_sSSSS-SS-SS_pPP_bBBB_gGGG
                                 zzで作成された注文でSSSS-SS-SSを開始日として期間PP,元金BBB,目標利益GGGで評価した結果が入るフォルダ
    stock_info.csv               全銘柄の概要情報ファイル
```
