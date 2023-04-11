import model            # 学習モデル
import argparse         # コマンドライン引数チェック用
import pandas as pd
import glob

def set_argparse():
    parser = argparse.ArgumentParser(description='TODO')
    parser.add_argument('dir', help='stock data directory')
    args = parser.parse_args()
    return args

def main():
    args = set_argparse()
    # ファイル読み込み
    files = glob.glob(args.dir + '/*.pkl')
    for file in files:
        df = pd.read_pickle(file)
        ret_df = model.estimate(df, 'day from 5 years ago', 'close')
        # 学習結果の保存
        ret_df.to_pickle('learning_data/' + str(df['code'].iloc[0]) + '.pkl')

if __name__ == "__main__":
    main()