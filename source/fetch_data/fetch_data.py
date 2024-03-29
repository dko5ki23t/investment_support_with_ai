from yahoo_finance_api2 import share
from yahoo_finance_api2.exceptions import YahooFinanceError
import pandas as pd
import argparse         # コマンドライン引数チェック用
from pathlib import Path
import os
import math
#import tkinter as tk
#import tkinter.messagebox as messagebox
import tqdm
import time

# 自作ロガー追加
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../logger'))
from logger import Logger
logger = Logger(__name__, 'fetch_data.log')


# 目的変数を作成する
def get_stockdata(code, period_type, period, freq_type, freq):
    company_code = str(code) + '.T'
    my_share = share.Share(company_code)
    symbol_data = None

    try:
        symbol_data = my_share.get_historical(period_type,
                                              period,
                                              freq_type,
                                              freq)
    except YahooFinanceError as e:
        logger.error(e.message)
        return None
    # 株価をデータフレームに入れている
    df_base = pd.DataFrame(symbol_data)
    df_base = pd.DataFrame(symbol_data.values(), index=symbol_data.keys()).T
    df_base.timestamp = pd.to_datetime(df_base.timestamp, unit='ms')
    df_base.index = pd.DatetimeIndex(df_base.timestamp, name='timestamp').tz_localize('UTC').tz_convert('Asia/Tokyo')
    # NaNを含む行は消す(TODO:これでいいのか)
    df_base = df_base.dropna(how='any')
    df_base = df_base.reset_index(drop=True)
    
    
    return df_base

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
    time_begin = time.perf_counter()
    for index in tqdm.tqdm(range(len(df))):
        item = df.iloc[index]
        stock_file_name = str(args.out_dir) + '/' + str(item['code']) + '.pkl'
        stock_df = pd.DataFrame
        # 既に株価情報ファイルが存在するか確認
        if os.path.exists(stock_file_name):     # 差分のみ取得
            stock_df = pd.read_pickle(stock_file_name)
            diff_days = math.ceil((pd.Timestamp.now() - stock_df['timestamp'].iloc[-1]).total_seconds() / 60 / 60 /24)
            ret = get_stockdata(item['code'], share.PERIOD_TYPE_DAY, diff_days, share.FREQUENCY_TYPE_DAY, 1)
            # 結合するが、日が変わっていないデータは捨てる
            stock_df = pd.concat([stock_df, ret], ignore_index=True)
            stock_df = stock_df.sort_values('timestamp')
            prev_date = pd.Timestamp(1900,1,1)
            drops = []
            for i in range(len(stock_df)):
                stamp = stock_df.iloc[i]['timestamp']
                if stamp.year == prev_date.year and stamp.month == prev_date.month and stamp.day == prev_date.day: # 同じ日
                    drops.append(stock_df.index[i - 1])
                prev_date = stamp
            stock_df = stock_df.drop(drops)
            stock_df.reset_index()
            stock_df['day'] = range(0, len(stock_df))
            stock_df['real/model'] = 'real'
            stock_df['code'] = item['code']
            stock_df['stock name'] = item['name']
        else:                                   # 全取得
            # エラーが出ない範囲で、最も長い期間で指定して取得する
            # TODO:マシな方法ない？
            for i in range(100, 0, -1):
                try:
                    ret = get_stockdata(
                        item['code'], share.PERIOD_TYPE_YEAR, i, share.FREQUENCY_TYPE_DAY, 1)    # i年前まで、1日ごとに取得
                except Exception as e:
                    continue
                #logger.info('get stock data (code:' + str(item['code']) + ') for ' + str(i) + ' years')
                stock_df = ret
                stock_df['day'] = range(0, len(stock_df))
                stock_df['real/model'] = 'real'
                stock_df['code'] = item['code']
                stock_df['stock name'] = item['name']
                break
        logger.info(stock_df)
        stock_df.to_pickle(stock_file_name)
    time_end = time.perf_counter()
    elapsed = time_end - time_begin
    logger.info('fetch complete in ' + str(elapsed) + 's')
    # 完了通知
    #tk.Tk().withdraw()
    #messagebox.showinfo('message', 'fetch complete')

if __name__ == "__main__":
    main()
