import argparse         # コマンドライン引数チェック用
import pandas as pd
import glob
from pathlib import Path
from yahoo_finance_api2 import share
from yahoo_finance_api2.exceptions import YahooFinanceError
import numpy as np
import tqdm
import datetime

# 自作ロガー追加
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../logger'))
from logger import Logger
logger = Logger(__name__, 'estimate.log')
# 自作モデル追加
sys.path.append(os.path.join(os.path.dirname(__file__), '../learn'))
import model


def set_argparse():
    parser = argparse.ArgumentParser(description='TODO')
    parser.add_argument('-d', '--dir', help='機械学習の統計データがあるディレクトリ', required=True)
    parser.add_argument('-t', '--term', help='何回後の終値開示までに', required=True)
    parser.add_argument('-n', '--now', help='現在使える資金', required=True)
    parser.add_argument('-g', '--gain', help='目標のプラス額', required=True)
    parser.add_argument('-a', '--analyze_dir', help='結果を保存するディレクトリ', required=True)
    args = parser.parse_args()
    return args

# とりあえずMSRが最も低い銘柄＆モデルの銘柄をおすすめする。(TODO)
def main():
    args = set_argparse()
    # ファイル読み込み
    files = glob.glob(args.dir + '/*.pkl')
    min_msr = 10000
    stock_name = ''
    predict_results = pd.DataFrame(columns=['code', 'name', 'timestamp', 'date',
                                            'model', 'price', 'term', 'predict',
                                            'predict gain', 'actual', 'msr'])
    print('scanning learning data ...')
    for index in tqdm.tqdm(range(len(files))):
        file = files[index]
        df = pd.read_pickle(file)
        # 銘柄情報出力
        #logger.info(str(df[0].code) + ':' + str(df[0].stock))
        #for model in df:
        #    (days, predict_vals) = model.predict(1)
        #    logger.info(str(model.name) + ' val:' + str(predict_vals[-1]) + ' msr:' + str(model.msr))
        #min_tmp = df['MSR'].min()
        #min_idx = df['MSR'].idxmin()
        # 現在の株価取得(TODO:最新とは言えなさそう？ & 値がnanになることあり)
        company_code = str(df[0].code) + '.T'
        my_share = share.Share(company_code)
        symbol_data = None

        # TODO:エラー発生時の対処として、リトライもあり？
        try:
            symbol_data = my_share.get_historical(share.PERIOD_TYPE_DAY,
                                                1,
                                                share.FREQUENCY_TYPE_MINUTE,
                                                1)
        except YahooFinanceError as e:
            logger.error(e.message)
            continue
        # symbol_dataがNoneの場合あり。それは無視する
        if symbol_data is None:
            continue
        df_now = pd.DataFrame(symbol_data.values(), index=symbol_data.keys()).T
        now_val = df_now['close'].iloc[-1]
        # TODO:より良いMSRを出したモデルを取得
        # とりあえず今はRNNのモデルを選択
        # TODO:エラー発生時には無視する
        for model in df:
            try:
                (days, predict_vals) = model.predict(1)
                tmp_s = pd.Series(
                    [model.code, model.stock, pd.Timestamp.now(), pd.Timestamp.now().date(),
                    model.name, now_val, int(args.term), predict_vals[-1],
                    (predict_vals[-1] - now_val), np.nan, model.msr],
                    index=predict_results.columns)
            except Exception as e:
                continue
            predict_results = pd.concat([predict_results, pd.DataFrame(data=tmp_s.values.reshape(1, -1), columns=predict_results.columns)])
        
        #if min_msr > min_tmp:
        #    min_msr = min_tmp
        #    stock_name = str(df['stock name'].iloc[0])
    # predict_resultsには全銘柄の予想値が入っている。モデルの予想的中率を知るために保存する。
    # 保存先ディレクトリがない場合は作成
    dir = Path(args.analyze_dir)
    dir.mkdir(parents=True, exist_ok=True)
    export_results = predict_results
    analyze_file_name = str(args.analyze_dir) + '/analyze_term_' + str(args.term) + '.pkl'
    # 既存のファイルがあればそのdataframeに追加
    # TODO:今のままだと同じ日に2回estimateすると2つのデータができてしまう
    if os.path.exists(analyze_file_name):
        past_results = pd.read_pickle(analyze_file_name)
        export_results = pd.concat([past_results, predict_results])
    export_results.to_pickle(analyze_file_name)

    # 引数で指定された額に収まる銘柄のみ抽出
    predict_results = predict_results[predict_results['price'] < int(args.now) / 100]
    # model3(RNN)のみ抽出
    predict_results = predict_results[predict_results['model'] == 'model3']
    # 予想損益の多い順に並べ替え
    predict_results = predict_results.sort_values('predict gain', ascending=False)
    logger.info(predict_results.head(10))
    print(predict_results.head(10))
    # 予想収益が多い順に、100株ずつ買っていく戦略
    print('Recommended purchase method:')
    balance = int(args.now)
    idx = 0
    while balance > 0:
        look = predict_results.iloc[idx]
        price = look['price'] * 100
        if price > balance:
            break
        idx += 1
        print(idx, look['code'], look['name'], 'Expected revenue:', (look['predict gain'] * 100))
        balance -= price
    

if __name__ == "__main__":
    main()