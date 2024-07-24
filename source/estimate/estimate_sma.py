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
import jpholiday

# 自作ロガー追加
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../logger'))
from logger import Logger
logger = Logger(__name__, 'estimate_sma.log')

input_directory_default = os.path.join(os.path.dirname(__file__), '../../db/stock_data')
out_estimate_directory_default = os.path.join(os.path.dirname(__file__), '../../db/estimates/estimate_sma')
stock_info_file_default = os.path.join(os.path.dirname(__file__), '../../db/stock_info.csv')

def name():
    """
    推定方法の名前
    """
    return 'SMA'

def version():
    """
    バージョン
    """
    return '1.0'

# 次の営業日を返す
def next_business_day(date: str):
    cur_date = datetime.datetime.strptime(date, "%Y-%m-%d")
    while True:
        cur_date = cur_date + datetime.timedelta(days=1)
        if cur_date.weekday() < 5 and not jpholiday.is_holiday(cur_date):
            return datetime.datetime.strftime(cur_date, "%Y-%m-%d")

def estimate(stock_df: pl.DataFrame, out_file: str, *discard):
    """
    SMA（単純移動平均）による推定を行う

    stock_df : DataFrame
                推定のもととなる株価データ
                必要な列 : 'Date', 'Code', 'Close'

    out_file : str
                推定結果の出力先ファイル名
    """

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
            out_day = next_business_day(out.get_column('Date')[i])
            # TODO: このへんの値適当なので、調整
            # 予想上昇幅
            gains = math.ceil(out.get_column('SMA5')[i] - out.get_column('SMA25')[i])
            # 信頼度スコア
            if gains == 0:
                score = 0
                gains = 1
            else:
                score = 1 / gains
            out_days.append({"date": out_day, "code": code, "gains": gains, "score": score, "yest_close": out.get_column('Close')[i]})
            isSMA5larger = True
        elif isSMA5larger and out.get_column('SMA5')[i] < out.get_column('SMA25')[i]:
            isSMA5larger = False
    output = {
        "code": code,
        "method_name": name(),
        "version": version(),
        "last_date": stock_df.get_column('Date')[-1],
        "estimate": out_days
    }
    # ファイル出力
    with open(out_file, 'w') as f:
        json.dump(output, f, indent=2)

def estimate_gen(input='', output='', stock_info_file='', filter_market_code=0):
    """
    【ジェネレータ】SMA（単純移動平均）による推定を行う

    input : str, default=''
            株価データが保存されたファイルまたはディレクトリ

    output : str, default=''
            推定結果の出力ファイル。ただし、inputの対象が1ファイルのときのみ有効

    stock_info_file : str, default=''
            全銘柄の情報（銘柄名や市場コード等）が記載されたCSVファイル

    filter_market_code : int, default=0
            市場コードによるフィルタ。0の場合はフィルタリングしない

    戻り値 : list
            index0 : 処理が終了したインデックス。 index1 : 処理総数
    """
    input_internal = input
    if input_internal == '':
        input_internal = input_directory_default
    # 銘柄情報ファイル読み込み
    stock_info_file_internal = stock_info_file
    if stock_info_file_internal == '':
        stock_info_file_internal = stock_info_file_default
    try:
        stock_info_df = pl.read_csv(stock_info_file_internal)
    except:
        print('銘柄情報データファイルの読み込みに失敗しました')
        sys.exit(1)
    if os.path.isdir(input_internal):
        data_files = glob.glob(input_internal + '/*.parquet')
    else:
        data_files = [input_internal]

    # 市場コードでフィルタリング
    filtered_stock_dfs = []
    for data_file in data_files:
        # 株価データ読み込み
        try:
            stock_df = pl.read_parquet(data_file)
        except:
            print('株価データファイルの読み込みに失敗しました')
            sys.exit(1)
        code = stock_df.get_column('Code')[0]
        market_code = stock_info_df.filter(pl.col('Code') == code).get_column('MarketCode')[0]
        if filter_market_code == 0 or market_code == filter_market_code:
            filtered_stock_dfs.append(stock_df)

    for index in tqdm.tqdm(range(len(filtered_stock_dfs))):
        stock_df = filtered_stock_dfs[index]
        code = stock_df.get_column('Code')[0]
        # 出力先ファイル決定
        out_estimate_file = output
        if out_estimate_file == '' or len(data_files) > 1:
            # 保存先ディレクトリがない場合は作成
            dir = Path(out_estimate_directory_default)
            dir.mkdir(parents=True, exist_ok=True)
            out_estimate_file = os.path.join(
                out_estimate_directory_default, code
            ) + '.json'

        estimate(stock_df, out_estimate_file)
        yield [index, len(filtered_stock_dfs)]

def set_argparse():
    parser = argparse.ArgumentParser(description='単純移動平均から予想を出す')
    parser.add_argument('input', help='株価データが保存されたファイルまたはディレクトリ')
    parser.add_argument('-m', '--filter_market_code', help='inputで指定したディレクトリ内の株価データを市場コードでフィルタリング', type=int, default=0)
    parser.add_argument('--stock_info', help='全銘柄の情報（銘柄名や市場コード等）が記載されたCSVファイル', default='')
    parser.add_argument('-o', '--output', help='予想の出力ファイル。ただし、inputの対象が1ファイルのときのみ有効', default='')
    args = parser.parse_args()
    return args

def main():
    args = set_argparse()
    for i in estimate_gen(args.input, args.output, args.stock_info, args.filter_market_code):
        pass

    sys.exit(0)


if __name__ == "__main__":
    main()
