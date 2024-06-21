import argparse         # コマンドライン引数チェック用
import pandas as pd
import numpy as np
import plotly.express as px

# 自作ロガー追加
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../source/logger'))
from logger import Logger
logger = Logger(__name__, 'visualize.log')
# 自作モデル追加
sys.path.append(os.path.join(os.path.dirname(__file__), '../source/learn'))
import model


def set_argparse():
    parser = argparse.ArgumentParser(description='TODO')
    parser.add_argument('outfile', help='出力ファイル名')
    parser.add_argument('-r', '--real', help='実際の株価データファイル(.pklファイル)', required=True)
    parser.add_argument('-l', '--learn', help='機械学習の統計データファイル(.pklファイル)', required=True)
    args = parser.parse_args()
    return args

def main():
    args = set_argparse()
    # ファイル読み込み
    df_real = pd.read_pickle(args.real)
    df_learn = pd.read_pickle(args.learn)
    # 実際の情報の日数分+10日間用意（TODO:引数で渡す）
    for model in df_learn:
        (days, vals) = model.predict(10)
        df_estimate = pd.DataFrame({'close':pd.Series(vals), 'day':pd.Series(days), 'real/model':model.name})
        df_real=pd.concat([df_real, df_estimate])

    fig = px.line(df_real, x='day', y='close', color='real/model')
    fig.write_html(args.outfile)

if __name__ == "__main__":
    main()