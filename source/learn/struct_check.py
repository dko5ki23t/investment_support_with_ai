from yahoo_finance_api2 import share
from yahoo_finance_api2.exceptions import YahooFinanceError
import pandas as pd
import argparse         # コマンドライン引数チェック用
from pathlib import Path
import os
import math
from plyer import notification
import tqdm
import glob
import model            # 学習モデル


def set_argparse():
    parser = argparse.ArgumentParser(description='TODO')
    parser.add_argument('fetched', help='name of file with stock code info')
    parser.add_argument('learning', help='株価データ保存先')
    parser.add_argument('estimate', help='株価データ保存先')
    args = parser.parse_args()
    return args

def main():
    args = set_argparse()
    # 株価情報データ
    files = glob.glob(args.fetched + '/*.pkl')
    file = files[0]
    df = pd.read_pickle(file)
    print('fetched dataframe')
    for column_name in df:
        print(column_name, type(df[column_name].iloc[0]))
    # 機械学習モデルデータ
    #files = glob.glob(args.learning + '/*.pkl')
    #file = files[0]
    #df = pd.read_pickle(file)
    #print('learning dataframe')
    #for column_name in df:
    #    print(column_name, type(df[column_name].iloc[0]))
    # 株価予実データ
    files = glob.glob(args.estimate + '/*.pkl')
    file = files[0]
    df = pd.read_pickle(file)
    print('estimate dataframe')
    for column_name in df:
        print(column_name, type(df[column_name].iloc[0]))

if __name__ == "__main__":
    main()
