import argparse         # コマンドライン引数チェック用
import pandas as pd
import glob
import numpy as np
import plotly.express as px

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
    x = np.arange(df_real['day from 5 years ago'].iloc[0], df_real['day from 5 years ago'].iloc[-1] + 10 + 1, 1)
    for index, item in df_learn.iterrows():
        y_hat = item['pipeline'].predict(x.reshape(-1, 1))
        df_estimate = pd.DataFrame({'close':pd.Series(y_hat), 'day from 5 years ago':pd.Series(x), 'real/estimate':item['model name']})
        df_real=pd.concat([df_real, df_estimate])

    fig = px.line(df_real, x='day from 5 years ago', y='close', color='real/model')
    fig.write_html(args.outfile)

if __name__ == "__main__":
    main()