import argparse         # コマンドライン引数チェック用
import pandas as pd
import glob

def set_argparse():
    parser = argparse.ArgumentParser(description='TODO')
    parser.add_argument('dir', help='機械学習の統計データがあるディレクトリ')
    parser.add_argument('term', help='何回後の終値開示までに')
    parser.add_argument('now', help='現在使える資金')
    parser.add_argument('gain', help='目標のプラス額')
    args = parser.parse_args()
    return args

# とりあえずMSRが最も低い銘柄＆モデルの銘柄をおすすめする。(TODO)
def main():
    args = set_argparse()
    # ファイル読み込み
    files = glob.glob(args.dir + '/*.pkl')
    min_msr = 10000
    stock_name = ''
    for file in files:
        df = pd.read_pickle(file)
        min_tmp = df['MSR'].min()
        print(df['stock name'].iloc[0])
        print(min_tmp)
        if min_msr > min_tmp:
            min_msr = min_tmp
            stock_name = str(df['stock name'].iloc[0])
    
    print('buy:' + stock_name)

if __name__ == "__main__":
    main()