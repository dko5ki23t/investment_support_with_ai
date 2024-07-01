import argparse         # コマンドライン引数チェック用
import json
import sys
import polars as pl
import os
import datetime
from pathlib import Path

# 自作ロガー追加
#import sys
#import os
#sys.path.append(os.path.join(os.path.dirname(__file__), '../logger'))
#from logger import Logger
#logger = Logger(__name__, 'analyze.log')

out_estimate_directory_default = os.path.join(os.path.dirname(__file__), '../../db/estimates/estimate_1')

def set_argparse():
    parser = argparse.ArgumentParser(description='単純移動平均から予想を出す')
    parser.add_argument('input', help='株価データが保存されたファイル')
    parser.add_argument('-o', '--output', help='予想の出力ファイル', default='')
    args = parser.parse_args()
    return args

def main():
    args = set_argparse()
    # 株価データ読み込み
    try:
        stock_df = pl.read_parquet(args.input)
    except:
        print('株価データファイルの読み込みに失敗しました')
        sys.exit(1)
    # 出力先ファイル決定
    out_strategy_file = args.output
    if out_strategy_file == '':
        # 保存先ディレクトリがない場合は作成
        dir = Path(out_estimate_directory_default)
        dir.mkdir(parents=True, exist_ok=True)
        out_strategy_file = os.path.join(
            out_estimate_directory_default, os.path.splitext(os.path.basename(args.input))[0]
        ) + '.json'

    start_idx = 25 # 単純移動平均が求められるところから
    # 5日単純移動平均を作る
    sma5 = stock_df.with_row_index(name="index")
    sma5 = sma5.rolling(index_column="index", period="5i").agg(
        [pl.mean('Close').alias('SMA5')]
    )['SMA5'].to_list()[start_idx:]
    # 25日単純移動平均を作る
    sma25 = stock_df.with_row_index(name="index")
    sma25 = sma25.rolling(index_column="index", period="25i").agg(
        [pl.mean('Close').alias('SMA25')]
    )['SMA25'].to_list()[start_idx:]
    # 連結
    dates = stock_df[start_idx:].get_column('Date').to_list()
    closes = stock_df[start_idx:].get_column('Close').to_list()
    out = pl.DataFrame({"Date": dates, "Close": closes, "SMA5": sma5, "SMA25": sma25})
    
    # SMA5がSMA25の値を上回り始める日を見つける
    out_days = []
    code = stock_df.get_column('Code')[0]
    isSMA5larger = True
    for i in range(len(out)):
        if not isSMA5larger and out.get_column('SMA5')[i] > out.get_column('SMA25')[i]:
            # 上回った次の日に注文する
            out_day = (
                datetime.datetime.strptime(out.get_column('Date')[i], "%Y-%m-%d") + datetime.timedelta(days=1)
            ).strftime("%Y-%m-%d")
            # 予想上昇幅
            gains = (out.get_column('SMA25')[i] - out.get_column('SMA25')[i-1]) * out.get_column('Close')[i]
            # 信頼度スコア
            # TODO: このへんの値適当なので、調整
            score = 1 / gains
            out_days.append({"date": out_day, "code": code, "gains": gains, "score": score})
            isSMA5larger = True
        elif isSMA5larger and out.get_column('SMA5')[i] < out.get_column('SMA25')[i]:
            isSMA5larger = False
    output = {"estimate": out_days}
    # ファイル出力
    with open(out_strategy_file, 'w') as f:
        json.dump(output, f, indent=2)


if __name__ == "__main__":
    main()
