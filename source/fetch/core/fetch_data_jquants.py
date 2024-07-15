import argparse
import sys
import json
import requests
import os
import polars as pl
from pathlib import Path
import tqdm
import time

sys.path.append(os.path.dirname(__file__))
import fetch_stock_info_jquants

# 自作ロガー追加
#import sys
#import os
#sys.path.append(os.path.join(os.path.dirname(__file__), '../logger'))
#from logger import Logger
#logger = Logger(__name__, 'fetch_data.log')

settings_file = os.path.join(os.path.dirname(__file__), '../../../settings/jquants.json')
input_file_default = os.path.join(os.path.dirname(__file__), '../../db/stock_info.csv')
out_dir_default = os.path.join(os.path.dirname(__file__), '../../db/stock_data')


# 日付の列追加
'''
def add_date(row):
    row['date'] = row['timestamp'].date()
    return row
'''
    
# 株価データを取得
def get_stock_data(code: str, id_token: str, date_from: str, date_to: str):
    headers = {'Authorization': 'Bearer {}'.format(id_token)}
    # TODO: fromとtoがあるとき
    url_str = f"https://api.jquants.com/v1/prices/daily_quotes?code={code}"
    # リクエスト送信
    try:
        r = requests.get(url_str, headers=headers)
        r_json = r.json()
    except:
        print('株価データ取得リクエスト中にエラーが発生しました')

    # ページングへの対処
    while "pagination_key" in r.json():
        pagination_key = r.json()["pagination_key"]
        r = requests.get(f"{url_str}&pagination_key={pagination_key}", headers=headers)
        r_json += r.json()
    
    return r_json
    
# 株価データを取得してテーブル作成、ファイルに保存
# ※新規で取得したデータが無い場合はファイル保存はしない
def get_stock_df(code: str, id_token: str, file_path: str):
    stock_df = pl.DataFrame
    # 既に株価データファイルが存在するか確認
    if os.path.exists(file_path):     # 存在するので、差分のみ取得
        # TODO
        pass
        '''
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
        '''
    else:        # 存在しないので、全取得
        try:
            stock_df = pl.from_dicts(get_stock_data(code, id_token, '', '')['daily_quotes'])
            stock_df.write_parquet(file_path)
        except Exception as e:
            print(f'株価データ取得に失敗しました（株コード：{code}）')
    return stock_df

def fetch_data(input: str, output: str, codes: list):
    try:
        settings = json.load(open(settings_file, 'r'))
    except:
        print('設定ファイルを読み込めませんでした')
        sys.exit(1)
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
    # IDトークン取得
    try:
        id_token = settings["id_token"]
    except:
        print('設定ファイルからIDトークンを取得できませんでした\nIDトークンを再取得します')
        id_token = fetch_stock_info_jquants.get_tokens(settings)
    
    # 各銘柄の株価データ取得
    print('各銘柄の株価データを取得しています・・・')
    if len(codes) == 0:
        for index in tqdm.tqdm(range(len(df))):
            code = df.get_column('Code')[index]
            file_path = out_dir + '/' + code + '.parquet'
            stock_df = get_stock_df(code, id_token, file_path)
    else:
        for index in tqdm.tqdm(range(len(codes))):
            code = codes[index]
            file_path = out_dir + '/' + code + '.parquet'
            stock_df = get_stock_df(code, id_token, file_path)
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

def fetch_data_gen(input: str, output: str, codes: list):
    try:
        settings = json.load(open(settings_file, 'r'))
    except:
        print('設定ファイルを読み込めませんでした')
        sys.exit(1)
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
    # IDトークン取得
    try:
        id_token = settings["id_token"]
    except:
        print('設定ファイルからIDトークンを取得できませんでした\nIDトークンを再取得します')
        id_token = fetch_stock_info_jquants.get_tokens(settings)
    
    # 各銘柄の株価データ取得
    print('各銘柄の株価データを取得しています・・・')
    if len(codes) == 0:
        for index in tqdm.tqdm(range(len(df))):
            code = df.get_column('Code')[index]
            file_path = out_dir + '/' + code + '.parquet'
            stock_df = get_stock_df(code, id_token, file_path)
            yield [index, len(df)]
    else:
        for index in tqdm.tqdm(range(len(codes))):
            code = codes[index]
            file_path = out_dir + '/' + code + '.parquet'
            stock_df = get_stock_df(code, id_token, file_path)
            yield [index, len(df)]
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
    parser = argparse.ArgumentParser(description='J-QuantsのAPIを用いて上場銘柄の株価データを取得する')
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
