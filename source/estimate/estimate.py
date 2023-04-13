import argparse         # コマンドライン引数チェック用
import pandas as pd
import glob

from yahoo_finance_api2 import share
from yahoo_finance_api2.exceptions import YahooFinanceError
import sys
import numpy as np

def set_argparse():
    parser = argparse.ArgumentParser(description='TODO')
    parser.add_argument('-d', '--dir', help='機械学習の統計データがあるディレクトリ', required=True)
    parser.add_argument('-t', '--term', help='何回後の終値開示までに', required=True)
    parser.add_argument('-n', '--now', help='現在使える資金', required=True)
    parser.add_argument('-g', '--gain', help='目標のプラス額', required=True)
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
        # 銘柄情報出力
        print(str(df['code'].iloc[0]) + ' : ' + str(df['stock name'].iloc[0]))
        # より良いMSRを出したモデルを取得
        min_tmp = df['MSR'].min()
        min_idx = df['MSR'].idxmin()
        model_pipeline = df.loc[min_idx]['pipeline']
        # 次の日～n日後の値推定(TODO:全日はいらない？)
        x = np.arange(df.loc[min_idx]['last_day'] + 1, df.loc[min_idx]['last_day'] + int(args.term) + 1, 1)
        predict_val = model_pipeline.predict(x.reshape(-1, 1))[-1]
        # 現在の株価取得(TODO:最新とは言えなさそう？ & 値がnanになることあり)
        company_code = str(df['code'].iloc[0]) + '.T'
        my_share = share.Share(company_code)
        symbol_data = None

        try:
            symbol_data = my_share.get_historical(share.PERIOD_TYPE_DAY,
                                                1,
                                                share.FREQUENCY_TYPE_MINUTE,
                                                1)
        except YahooFinanceError as e:
            print(e.message)
            sys.exit(1)
        # symbol_dataがNoneの場合あり。それは無視する
        if symbol_data is None:
            continue
        df_now = pd.DataFrame(symbol_data.values(), index=symbol_data.keys()).T
        print('now:' + str(df_now['close'].iloc[-1]) + ' predict:' + str(predict_val))
        
        if min_msr > min_tmp:
            min_msr = min_tmp
            stock_name = str(df['stock name'].iloc[0])
    
    print('buy:' + stock_name)

if __name__ == "__main__":
    main()