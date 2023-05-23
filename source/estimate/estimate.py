import argparse         # コマンドライン引数チェック用
import pandas as pd
import glob
from pathlib import Path
from yahoo_finance_api2 import share
from yahoo_finance_api2.exceptions import YahooFinanceError
import numpy as np
import tqdm
import pickle

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
    parser.add_argument('-t', '--term', help='何回後の終値開示までに', type=int, required=True)
    parser.add_argument('-n', '--now', help='現在使える資金', required=True)
    parser.add_argument('-g', '--gain', help='目標のプラス額', required=True)
    parser.add_argument('-a', '--analyze_dir', help='結果を保存するディレクトリ', required=True)
    parser.add_argument('-u', '--use_existing', help='株価を実行中に取得せず、既存データを使う', action='store_true')
    parser.add_argument('-f', '--fetched_dir', help='株価の既存データ')
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
    
    fetched_df_dict = {}
    if args.use_existing == True:   # 現在の株価を既存データから取得
        # 株価データファイル読み込み
        fetched_files = glob.glob(args.fetched_dir + '/*.pkl')
        # 日経平均株価は除外
        fetched_files = [e for e in fetched_files if not e.endswith('N225.pkl')]
        # 銘柄コードとデータフレームのdictにする
        print('(1/2)read and construct fetched data ...')
        for i in tqdm.tqdm(range(len(fetched_files))):
            fetched_file = fetched_files[i]
            fetched_df = pd.read_pickle(fetched_file)
            fetched_df_dict[int(fetched_df['code'].iloc[0])] = fetched_df
    
    print('scanning learning data ...')
    for index in tqdm.tqdm(range(len(files))):
        file = files[index]
        f = open(file, 'rb')
        models = pickle.load(f)
        f.close
#        df = pd.read_pickle(file)
        meta_data = models.pop(0)
        # 銘柄情報出力
        #logger.info(str(df[0].code) + ':' + str(df[0].stock))
        #for model in df:
        #    (days, predict_vals) = model.predict(1)
        #    logger.info(str(model.name) + ' val:' + str(predict_vals[-1]) + ' msr:' + str(model.msr))
        #min_tmp = df['MSR'].min()
        #min_idx = df['MSR'].idxmin()

        now_val = float()
        if args.use_existing == True:   # 現在の株価を既存データから取得
            fetched_df = fetched_df_dict[meta_data['code']]
            now_val = fetched_df['close'].iloc[-1]      # 最新の終値を現在の株価とする
        else:
            # 現在の株価取得(TODO:最新とは言えなさそう？ & 値がnanになることあり)
            company_code = str(meta_data['code']) + '.T'
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
                logger.info('cannot get current price of ' + company_code)
                continue
            df_now = pd.DataFrame(symbol_data.values(), index=symbol_data.keys()).T
            now_val = df_now['close'].iloc[-1]
        # TODO:より良いMSRを出したモデルを取得
        # とりあえず今はRNNのモデルを選択
        # TODO:エラー発生時には無視する
        for model in models:
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