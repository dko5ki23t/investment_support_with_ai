import argparse         # コマンドライン引数チェック用
import json
import sys
import polars as pl
import os
import datetime
from pathlib import Path
import math
import glob
import tqdm

# 自作ロガー追加
#import sys
#import os
#sys.path.append(os.path.join(os.path.dirname(__file__), '../logger'))
#from logger import Logger
#logger = Logger(__name__, 'analyze.log')

input_directory_default = os.path.join(os.path.dirname(__file__), '../../db/stock_data')
out_estimate_directory_default = os.path.join(os.path.dirname(__file__), '../../db/estimates/estimate_1')

def estimate(stock_df: pl.DataFrame, out_file: str):
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
        if not isSMA5larger and out.get_column('SMA5')[i] > out.get_column('SMA5')[i-1]:
            # 上回った次の日に注文する
            out_day = (
                datetime.datetime.strptime(out.get_column('Date')[i], "%Y-%m-%d") + datetime.timedelta(days=1)
            ).strftime("%Y-%m-%d")
            # TODO: このへんの値適当なので、調整
            # 予想上昇幅
            gains = math.ceil(out.get_column('SMA5')[i] - out.get_column('SMA25')[i])
            # 信頼度スコア
            if gains == 0:
                score = 0
                gains = 1
            else:
                score = 1 / gains
            out_days.append({"date": out_day, "code": code, "gains": gains, "score": score})
            isSMA5larger = True
        elif isSMA5larger and out.get_column('SMA5')[i] < out.get_column('SMA25')[i]:
            isSMA5larger = False
    output = {"estimate": out_days}
    # ファイル出力
    with open(out_file, 'w') as f:
        json.dump(output, f, indent=2)

def estimate_gen(input: str, output: str):
    input_internal = input
    if input_internal == '':
        input_internal = input_directory_default
    if os.path.isdir(input_internal):
        data_files = glob.glob(input_internal + '/*.parquet')
    else:
        data_files = [input_internal]

    for index in tqdm.tqdm(range(len(data_files))):
        data_file = data_files[index]
        # 株価データ読み込み
        try:
            stock_df = pl.read_parquet(data_file)
        except:
            print('株価データファイルの読み込みに失敗しました')
            sys.exit(1)
        # 出力先ファイル決定
        out_strategy_file = output
        if out_strategy_file == '' or len(data_files) > 1:
            # 保存先ディレクトリがない場合は作成
            dir = Path(out_estimate_directory_default)
            dir.mkdir(parents=True, exist_ok=True)
            out_strategy_file = os.path.join(
                out_estimate_directory_default, os.path.splitext(os.path.basename(data_file))[0]
            ) + '.json'

        estimate(stock_df, out_strategy_file)
        yield [index, len(data_files)]

def set_argparse():
    parser = argparse.ArgumentParser(description='単純移動平均から予想を出す')
    parser.add_argument('input', help='株価データが保存されたファイルまたはディレクトリ')
    parser.add_argument('-o', '--output', help='予想の出力ファイル。ただし、inputの対象が1ファイルのときのみ有効', default='')
    args = parser.parse_args()
    return args

def main():
    args = set_argparse()
    data_files = []
    if os.path.isdir(args.input):
        data_files = glob.glob(args.input + '/*.parquet')
    else:
        data_files = [args.input]

    for index in tqdm.tqdm(range(len(data_files))):
        data_file = data_files[index]
        # 株価データ読み込み
        try:
            stock_df = pl.read_parquet(data_file)
        except:
            print('株価データファイルの読み込みに失敗しました')
            sys.exit(1)
        # 出力先ファイル決定
        out_strategy_file = args.output
        if out_strategy_file == '' or len(data_files) > 1:
            # 保存先ディレクトリがない場合は作成
            dir = Path(out_estimate_directory_default)
            dir.mkdir(parents=True, exist_ok=True)
            out_strategy_file = os.path.join(
                out_estimate_directory_default, os.path.splitext(os.path.basename(data_file))[0]
            ) + '.json'

        estimate(stock_df, out_strategy_file)

    sys.exit(0)


if __name__ == "__main__":
    main()
