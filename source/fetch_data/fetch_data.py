from yahoo_finance_api2 import share
from yahoo_finance_api2.exceptions import YahooFinanceError
import pandas as pd
import sys
import plotly.express as px
from numpy.polynomial import Polynomial
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
import argparse         # コマンドライン引数チェック用
from pathlib import Path

# 目的変数を作成する
def kabuka(code, year, day):
    company_code = str(code) + '.T'
    my_share = share.Share(company_code)
    symbol_data = None

    try:
        symbol_data = my_share.get_historical(share.PERIOD_TYPE_YEAR,
                                              year,
                                              share.FREQUENCY_TYPE_DAY,
                                              day)
    except YahooFinanceError as e:
        print(e.message)
        sys.exit(1)
    # 株価をデータフレームに入れている
    df_base = pd.DataFrame(symbol_data)
    df_base = pd.DataFrame(symbol_data.values(), index=symbol_data.keys()).T
    df_base.timestamp = pd.to_datetime(df_base.timestamp, unit='ms')
    df_base.index = pd.DatetimeIndex(df_base.timestamp, name='timestamp').tz_localize('UTC').tz_convert('Asia/Tokyo')
    # NaNを含む行は消す(TODO:これでいいのか)
    df_base = df_base.dropna(how='any')
    df_base = df_base.reset_index(drop=True)
    
    
    return company_code, df_base

def set_argparse():
    parser = argparse.ArgumentParser(description='TODO')
    parser.add_argument('filename', help='name of file with stock code info')
    parser.add_argument('out_dir', help='株価データ保存先')
    args = parser.parse_args()
    return args

def main():
    args = set_argparse()
    # ファイル読み込み
    df = pd.read_csv(args.filename)
    # データ取得&バイナリにして保存
    # ディレクトリがない場合は作成
    dir = Path(args.out_dir)
    dir.mkdir(parents=True, exist_ok=True)
    for index, item in df.iterrows():
        ret = kabuka(item['code'], 5, 1)    # 5年前まで、1日ごとに取得
        stock_df = ret[1]
        stock_df['day from 5 years ago'] = range(0, len(stock_df))
        stock_df['real/model'] = 'real'
        stock_df['code'] = item['code']
        stock_df['stock name'] = item['name']
        stock_df.to_pickle(str(args.out_dir) + '/' + str(item['code']) + '.pkl')
        # TODO:ログ保存

if __name__ == "__main__":
    main()
