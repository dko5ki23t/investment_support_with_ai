from yahoo_finance_api2 import share
from yahoo_finance_api2.exceptions import YahooFinanceError
import pandas as pd
import argparse
from pathlib import Path
import os
import tqdm
import time
from plyer import notification
import datetime

# 自作ロガー追加
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../logger'))
from logger import Logger
logger = Logger(__name__, 'fetch_data.log')


# 日付の列追加
def add_date(row):
    row['date'] = row['timestamp'].date()
    return row

# 株価情報を取得
def get_stockdata(company_code, period_type, period, freq_type, freq):
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
    df_base = pd.DataFrame(symbol_data.values(), index=symbol_data.keys()).T
    df_base.timestamp = pd.to_datetime(df_base.timestamp, unit='ms')
    df_base.index = pd.DatetimeIndex(df_base.timestamp, name='timestamp').tz_localize('UTC').tz_convert('Asia/Tokyo')
    # NaNを含む行は消す(TODO:これでいいのか)
    df_base = df_base.dropna(how='any')
    df_base = df_base.reset_index(drop=True)
    # 日付列追加
    df_base = df_base.apply(add_date, axis=1)
    # (明示的に)数値に変換
    df_base = df_base.astype({'open':float, 'high':float, 'low':float, 'close':float, 'volume':float})
    # 'date': datetime.dateもしたかったが、dfにdateはない(datetime64[ns]ならある)からやめといた

    return df_base

# 株価情報データを作成
def build_stock_df(code:str, code_real:str, name:str, file:str):
    stock_df = pd.DataFrame
    # 既に株価情報ファイルが存在するか確認
    if os.path.exists(file):     # 存在するので、差分のみ取得
        stock_df = pd.read_pickle(file)
        diff_days = (pd.Timestamp.now().date() - stock_df['date'].iloc[-1]).days + 1  # 同日でも、0にはしない
        ret = get_stockdata(code_real, share.PERIOD_TYPE_DAY, diff_days, share.FREQUENCY_TYPE_DAY, 1)
        # 結合するが、日が変わっていないデータは古い方を捨てる
        stock_df = pd.concat([stock_df, ret], ignore_index=True)
        stock_df = stock_df.sort_values('timestamp')
        prev_date = pd.Timestamp(1900,1,1).date()
        drops = []
        for i in range(len(stock_df)):
            stamp = stock_df.iloc[i]['date']
            if stamp == prev_date: # 同じ日
                drops.append(stock_df.index[i - 1])
            prev_date = stamp
        stock_df = stock_df.drop(drops)
        stock_df.reset_index()
        stock_df['day'] = range(0, len(stock_df))
        stock_df['real/model'] = 'real'
        stock_df['code'] = code
        stock_df['stock name'] = name
    else:        # 存在しないので、全取得
        # エラーが出ない範囲で、最も長い期間で指定して取得する(MAX100年)
        # TODO:マシな方法ない？
        for i in range(100, 0, -1):
            try:
                ret = get_stockdata(
                    code_real, share.PERIOD_TYPE_YEAR, i, share.FREQUENCY_TYPE_DAY, 1)    # i年前まで、1日ごとに取得
            except Exception as e:
                continue
            stock_df = ret
            stock_df['day'] = range(0, len(stock_df))
            stock_df['real/model'] = 'real'
            stock_df['code'] = code
            stock_df['stock name'] = name
            break
    return stock_df

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
    # 日経平均株価取得
    print('(1/2)fetch Nikkei225 data...')
    code = 'N225'
    code_real = '^N225'
    name = '日経平均株価'
    file = args.out_dir + '/N225.pkl'
    stock_df = build_stock_df(code, code_real, name, file)
    stock_df.to_pickle(file)
    print('done')
    # 各銘柄取得
    print('(2/2)fetch each stock data...')
    for index in tqdm.tqdm(range(len(df))):
        item = df.iloc[index]
        code = str(item['code'])
        code_real = code + '.T'
        name = item['name']
        file = args.out_dir + '/' + code + '.pkl'
        stock_df = build_stock_df(code, code_real, name, file)
        stock_df.to_pickle(file)
    print('done')
    time_end = time.perf_counter()
    elapsed = time_end - time_begin
    logger.info('fetch and save all data in ' + str(elapsed) + 's')
    # 完了通知
    notification.notify(
        title="complete fetching data",
        message="complete fetching data",
        app_name="fetch_data.py",
        timeout=10
    )

if __name__ == "__main__":
    main()
