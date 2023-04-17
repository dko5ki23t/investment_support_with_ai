import argparse         # コマンドライン引数チェック用
import pandas as pd
import numpy as np
import plotly.express as px

# 自作ロガー追加
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../source/logger'))
from logger import Logger
logger = Logger(__name__, '../log', 'visualize.log')


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
        if item['opt1'] is None:
            y_hat = item['pipeline'].predict(x.reshape(-1, 1))
        # TODO:もうちょいうまくやる
        else:
            logger.info('rnn')
            test_data = item['opt1'][1]
            x_test = []
            window_size = 60
            for i in range(window_size, len(x)-11):
                x_test.append(test_data[i-window_size:i, 0])
            x_test = np.array(x_test)
            x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
            predictions = item['pipeline'].predict(x_test)
            y_hat = item['opt1'][0].inverse_transform(predictions)
            # 1次元化
            y_hat = y_hat.ravel()
        df_estimate = pd.DataFrame({'close':pd.Series(y_hat), 'day from 5 years ago':pd.Series(x), 'real/model':item['model name']})
        df_real=pd.concat([df_real, df_estimate])

    fig = px.line(df_real, x='day from 5 years ago', y='close', color='real/model')
    fig.write_html(args.outfile)

if __name__ == "__main__":
    main()