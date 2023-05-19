import argparse         # コマンドライン引数チェック用
import pandas as pd
import glob
from pathlib import Path
import numpy as np
import tqdm

# 自作ロガー追加
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../logger'))
from logger import Logger
logger = Logger(__name__, 'analyze.log')

def set_argparse():
    parser = argparse.ArgumentParser(description='TODO')
    parser.add_argument('-f', '--fetched_dir', help='株価データが保存されたディレクトリ', required=True)
    parser.add_argument('-a', '--analyze_dir', help='株価予測を保存したディレクトリ', required=True)
    args = parser.parse_args()
    return args

def set_actual(row, fetched_df_dict):
    if np.isnan(row['actual']):
        fetched_df = fetched_df_dict[row['code']]
        # 対応する日付の実測データがあるか判定
        same_date_df = fetched_df['date'] == row['date']
        if same_date_df.values.sum() > 0:
            # 実測データ格納
            row['actual'] = fetched_df[same_date_df]['close'].iloc[-1]
            row['delta value'] = row['actual'] - row['predict']
            row['delta ration'] = row['actual'] / row['predict']
    return row


def main():
    args = set_argparse()
    # 株価データファイル読み込み
    fetched_files = glob.glob(args.fetched_dir + '/*.pkl')
    # 銘柄コードとデータフレームのdictにする
    fetched_df_dict = {}
    print('(1/2)read and construct fetched data ...')
    for i in tqdm.tqdm(range(len(fetched_files))):
        fetched_file = fetched_files[i]
        fetched_df = pd.read_pickle(fetched_file)
        fetched_df_dict[int(fetched_df['code'].iloc[0])] = fetched_df
    # 株価予測ファイル読み込み
    analyze_files = glob.glob(args.analyze_dir + '/*.pkl')
    # 株価予測データフレームの結果未格納部分に結果を格納する
    print('(2/2)analyze learning data ...')
    for i in tqdm.tqdm(range(len(analyze_files))):
        analyze_file = analyze_files[i]
        analyze_df = pd.read_pickle(analyze_file)
        # actual列を追加
        if 'actual' not in analyze_df.columns:
            analyze_df['actual'] = np.nan
        analyze_df = analyze_df.apply(set_actual, args=(fetched_df_dict,), axis=1)
        # 出力
        analyze_df.to_pickle(analyze_file)
        logger.info(analyze_df)

if __name__ == "__main__":
    main()
