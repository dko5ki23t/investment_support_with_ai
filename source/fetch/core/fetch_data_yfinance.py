import argparse
import sys
import os
import polars as pl
from pathlib import Path
import tqdm
import time
import yfinance as yf
from datetime import datetime

sys.path.append(os.path.dirname(__file__))
import fetch_stock_info_jquants

# 自作ロガー追加
#import sys
#import os
#sys.path.append(os.path.join(os.path.dirname(__file__), '../logger'))
#from logger import Logger
#logger = Logger(__name__, 'fetch_data.log')

input_file_default = os.path.join(os.path.dirname(__file__), '../../db/stock_info.csv')
out_dir_default = os.path.join(os.path.dirname(__file__), '../../db/stock_data')


# 株価データを取得
def get_stock_data(code: str, date_from: str, date_to: str):
    end = date_to
    if end == '':
        end = datetime.today().strftime("%Y-%m-%d")
    # リクエスト送信
    try:
        if date_from != '':
            r = yf.download(code, start=date_from, end=end)
        else:
            r = yf.download(code)
    except:
        print('株価データ取得リクエスト中にエラーが発生しました')
    
    return r
    
# 株価データを取得してテーブル作成、ファイルに保存
# ※新規で取得したデータが無い場合はファイル保存はしない
def get_stock_df(code: str, file_path: str):
    # 5桁コード->ティッカーに
    ticker = code
    if len(ticker) > 4 and ticker[4] == '0':
        ticker = ticker[:4]
    ticker = ticker + '.T'
    stock_df = pl.DataFrame
    # 既に株価データファイルが存在するか確認
    if os.path.exists(file_path):     # 存在するので、差分のみ取得
        try:
            stock_df = pl.read_parquet(file_path)
            # 最後に取得した日付以降の株価データを取得する
            start = stock_df.get_column('Date')[-1]
            # 最後に取得した日付の行は削除
            stock_df_removed = stock_df.filter(pl.col('Date') != start)
            # 実際にデータを取得
            ret = pl.from_pandas(get_stock_data(ticker, start, '').reset_index())
            ret = ret.with_columns(ret['Date'].dt.strftime("%Y-%m-%d"))
            # Code列追加
            stock_df = stock_df.with_columns(pl.lit(code).alias("Code"))
            stock_df = pl.concat([stock_df_removed, ret])
            ## 株価データをファイルに書き込み
            stock_df.write_parquet(file_path)
        except:
            print(f'株価データ取得に失敗しました（株コード：{code}）')
    else:        # 存在しないので、全取得
        try:
            # 取得したpandasのDataFrameの、インデックス列(Date)を列として追加してPolarsのDataFrameに変換
            stock_df = pl.from_pandas(get_stock_data(ticker, '', '').reset_index())
            # Date列をdatetime->strに変換
            stock_df = stock_df.with_columns(stock_df['Date'].dt.strftime("%Y-%m-%d"))
            # Code列追加
            stock_df = stock_df.with_columns(pl.lit(code).alias("Code"))
            # 株価データをファイルに書き込み
            stock_df.write_parquet(file_path)
        except Exception as e:
            print(f'株価データ取得に失敗しました（株コード：{code}）')
    return stock_df

def fetch_data(input: str, output: str, codes: list):
    # 入力ファイル決定
    input_file = input
    if input_file == '':
        input_file = input_file_default
    # 出力先ディレクトリ決定
    out_dir = output
    if out_dir == '':
        out_dir = out_dir_default

    # 銘柄情報ファイル読み込み
    df = pl.read_csv(input_file)
    # 保存先ディレクトリがない場合は作成
    dir = Path(out_dir)
    dir.mkdir(parents=True, exist_ok=True)
    # 取得開始時刻
    time_begin = time.perf_counter()
    '''
    # 日経平均株価取得
    print('(1/2)fetch Nikkei225 data...')
    code = 'N225'
    code_real = '^N225'
    name = '日経平均株価'
    file = args.out_dir + '/N225.pkl'
    stock_df = build_stock_df(code, code_real, name, file)
    stock_df.to_pickle(file)
    print('done')
    '''
    
    # 各銘柄の株価データ取得
    print('各銘柄の株価データを取得しています・・・')
    if codes is None or len(codes) == 0:
        for index in tqdm.tqdm(range(len(df))):
            code = df.get_column('Code')[index]
            file_path = out_dir + '/' + code + '.parquet'
            stock_df = get_stock_df(code, file_path)
    else:
        for index in tqdm.tqdm(range(len(codes))):
            code = codes[index]
            file_path = out_dir + '/' + code + '.parquet'
            stock_df = get_stock_df(code, file_path)
    print('完了')
    time_end = time.perf_counter()
    elapsed = time_end - time_begin
    #logger.info('fetch and save all data in ' + str(elapsed) + 's')
    # 完了通知
    '''
    notification.notify(
        title="complete fetching data",
        message="complete fetching data",
        app_name="fetch_data.py",
        timeout=10
    )
    '''

def set_argparse():
    parser = argparse.ArgumentParser(description='Yahoo FinanceのAPIを用いて上場銘柄の株価データを取得する')
    parser.add_argument('-i', '--input', help='銘柄情報が記載されたCSVファイル', default='')
    parser.add_argument('-o', '--output', help='株価データ保存先ディレクトリ', default='')
    args = parser.parse_args()
    return args

def main():
    args = set_argparse()
    fetch_data(args.input, args.output)

if __name__ == "__main__":
    main()
    sys.exit()
