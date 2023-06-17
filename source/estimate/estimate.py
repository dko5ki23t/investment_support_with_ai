import argparse         # コマンドライン引数チェック用
import pandas as pd
import glob
from pathlib import Path
from yahoo_finance_api2 import share
from yahoo_finance_api2.exceptions import YahooFinanceError
import numpy as np
import tqdm
import pickle
import time
from plyer import notification
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
    parser.add_argument('-t', '--term', help='何回後の終値開示までに', type=int, required=True)
    parser.add_argument('-n', '--now', help='現在使える資金', required=True)
    parser.add_argument('-g', '--gain', help='目標のプラス額', required=True)
    parser.add_argument('-a', '--analyze_dir', help='結果を保存するディレクトリ', required=True)
    parser.add_argument('-u', '--use_existing', help='株価を実行中に取得せず、既存データを使う', action='store_true')
    parser.add_argument('-f', '--fetched_dir', help='株価の既存データ', required=True)
    args = parser.parse_args()
    return args

# とりあえずMSRが最も低い銘柄＆モデルの銘柄をおすすめする。(TODO)
def main():
    args = set_argparse()
    # ファイル読み込み
    files = glob.glob(args.dir + '/*.pkl')
    files = [e for e in files if not e.endswith('global.pkl')]
    global_file = args.dir + '/global.pkl'
    min_msr = 10000
    predict_results = pd.DataFrame(columns=['code', 'name', 'timestamp', 'date',
                                            'model', 'price', 'term', 'predict',
                                            'predict gain', 'actual', 'msr'])
    
    time_begin = time.perf_counter()
    fetched_df_dict = {}
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
    
    analyze_file_name = str(args.analyze_dir) + '/analyze_term_' + str(args.term) + '.pkl'
    # 既存のファイルがあり、次の日の予測値があればそちらを抽出して使う
    #if os.path.exists(analyze_file_name):
    #    past_results = pd.read_pickle(analyze_file_name)
    #    already_s = past_results['date'] == datetime.date.today() + datetime.timedelta(days=1)   # TODO
    #    # 次の日の予測値がある（既に予測している）行のみ抽出
    #    predict_results = past_results[already_s]

    # 既存ファイルが無いか、あっても次の日の予測値がない場合は予測開始
    if len(predict_results) == 0:
        #############################################
        # 全銘柄をinputとする学習モデルの予測
        #############################################
        # TODO:名前、globalというかacross??
        print('(2/2)scan learning data and estimate ...')
        f = open(global_file, 'rb')
        global_models = pickle.load(f)
        f.close
        meta_data = global_models.pop(0)
        for model in global_models:
            try:
                codes = model.columns
                (days, predict_vals) = model.predict(1)
                for i in range(predict_vals.shape[-1]):
                    code = int(codes[i])
                    fetched_df = fetched_df_dict[code]
                    name = fetched_df['stock name'].iloc[0]
                    # 現在の株価取得。各銘柄inputにも同じコードがあるため、
                    # TODO: コードをまとめる
                    now_val = float()
                    if args.use_existing == True:   # どの株も、現在の株価を既存データから取得(Web API使わない)
                        now_val = fetched_df['close'].iloc[-1]      # 最新の終値を現在の株価とする
                    else:
                        # 現在の株価取得(TODO:最新とは言えなさそう？ & 値がnanになることあり)
                        company_code = str(code) + '.T'
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
                        # symbol_dataがNoneの場合あり
                        # その場合は最新の終値を現在の株価とする
                        if symbol_data is None:
                            logger.info('cannot get current price of ' + company_code)
                            now_val = fetched_df['close'].iloc[-1]      # 最新の終値を現在の株価とする
                        else:
                            df_now = pd.DataFrame(symbol_data.values(), index=symbol_data.keys()).T
                            now_val = df_now['close'].iloc[-1]
                    # 各銘柄の行を作成
                    predict_val = predict_vals[-1, i]
                    logger.info('predict_val:' + str(predict_val))
                    tmp_s = pd.Series(
                        [code, name, pd.Timestamp.now(), meta_data['last_date'] + datetime.timedelta(days=1),    # TODO:暫定で1日後
                        model.name, now_val, int(args.term), predict_val,
                        (predict_val - now_val), np.nan, model.msr],
                        index=predict_results.columns)
            except Exception as e:
                logger.error('[' + str(model.name) + '] :')
                logger.error(e)
                continue
            predict_results = pd.concat([predict_results, pd.DataFrame(data=tmp_s.values.reshape(1, -1), columns=predict_results.columns)])



        #############################################
        # 各銘柄をinputとする学習モデルの予測
        #############################################
        print('(2/2)scan learning data and estimate ...')
        for index in tqdm.tqdm(range(len(files))):
            file = files[index]
            f = open(file, 'rb')
            models = pickle.load(f)
            f.close
            meta_data = models.pop(0)

            now_val = float()
            if args.use_existing == True:   # どの株も、現在の株価を既存データから取得(Web API使わない)
                fetched_df = fetched_df_dict[int(meta_data['code'])]
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
                # symbol_dataがNoneの場合あり
                # その場合は最新の終値を現在の株価とする
                if symbol_data is None:
                    logger.info('cannot get current price of ' + company_code)
                    # TODO:原因見つけて直す
                    try:
                        fetched_df = fetched_df_dict[int(meta_data['code'])]
                    except Exception as e:
                        logger.info('TODO:原因見つけて直す')
                        logger.error(e)
                        continue
                    now_val = fetched_df['close'].iloc[-1]      # 最新の終値を現在の株価とする
                else:
                    df_now = pd.DataFrame(symbol_data.values(), index=symbol_data.keys()).T
                    now_val = df_now['close'].iloc[-1]
            # TODO:より良いMSRを出したモデルを取得
            # とりあえず今はRNNのモデルを選択
            # TODO:エラー発生時には無視する
            for model in models:
                try:
                    (days, predict_vals) = model.predict(1)
                    tmp_s = pd.Series(
                        [model.code, model.stock, pd.Timestamp.now(), meta_data['last_date'].date() + datetime.timedelta(days=1),    # TODO:暫定で1日後
                        model.name, now_val, int(args.term), predict_vals[-1],
                        (predict_vals[-1] - now_val), np.nan, model.msr],
                        index=predict_results.columns)
                except Exception as e:
                    logger.error('[' + str(model.code) + '] ' + str(model.name) + ':')
                    logger.error(e)
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
    if os.path.exists(analyze_file_name):
        past_results = pd.read_pickle(analyze_file_name)
        export_results = pd.concat([past_results, predict_results])
        export_results = export_results.drop_duplicates(subset=['date', 'model', 'code'], keep='last')
    export_results.to_pickle(analyze_file_name)

    # 引数で指定された額に収まる銘柄のみ抽出
    predict_results = predict_results[predict_results['price'] < int(args.now) / 100]
    
    #################################################
    # TODO: １,２の予測方法を、コメントで切り替えてる
    # 予測方法１
    #################################################

    # model3(RNN)のみ抽出
    final_results = predict_results[predict_results['model'] == 'model3']
    # model4(RNN)のみ抽出
    #final_results = predict_results[predict_results['model'] == 'model5']
    # model6(RNN)のみ抽出
    #final_results = predict_results[predict_results['model'] == 'model6']

    # 予想損益の多い順に並べ替え
    final_results = final_results.sort_values('predict gain', ascending=False)

    #################################################
    # 予測方法２
    #################################################
    '''
    final_results = pd.DataFrame(columns=predict_results.columns)
    # 統計情報をもとに、当たる確率が高いモデルの予測のみ抽出
    summary_file_name = str(args.analyze_dir) + '/summary_analyze_term_' + str(args.term) + '.pkl'
    if os.path.exists(summary_file_name):
        summary_df = pd.read_pickle(summary_file_name)
        summary_df = summary_df.reset_index()
        files = glob.glob(args.dir + '/*.pkl')
        for key in fetched_df_dict.keys():  # (int)銘柄コードでループ
            s_key = str(key)
            sp_df = summary_df.query('code == @s_key')
            tmp_s = predict_results[predict_results['code'] == s_key]
            if (len(sp_df) > 0):
                model_name = sp_df.loc[sp_df['delta ration msr'].idxmin(), 'model']
                # query使うと、tmp_sを書き換えるときに'A value is trying to be set on a copy of a slice from a DataFrame'の警告が出る
                # 原因不明。最新のpandasだと出ないかも。
                #tmp_s = predict_results.query('code == @s_key and model == @model_name')
                tmp_s = tmp_s[tmp_s['model'] == model_name]
            else:
                #tmp_s = predict_results.query('code == @s_key and model == model3')
                tmp_s = tmp_s[tmp_s['model'] == 'model3']
            # 対象が見つかっていれば追加
            if len(tmp_s) > 0:
                # TODO:一旦、MSRの列を、今までの統計から得られるMSRに書き換える
                tmp_s.loc[:, 'msr'] = sp_df['delta ration msr'].min()
                final_results = pd.concat([final_results, pd.DataFrame(data=tmp_s.values.reshape(1, -1), columns=predict_results.columns)])
    else:   # なければmodel3のみ抽出
        final_results = predict_results[predict_results['model'] == 'model3']
    
    # MSR、予想損益の多い順に並べ替え
    final_results = final_results.sort_values(['msr', 'predict gain'], ascending=[True, False])
    
    # MSRが大きいほど、予測損益が0に近づくような値をdataframeに追加
    final_results['ref val'] = (1.0 - final_results['msr']) * final_results['predict gain']
    final_results = final_results.sort_values(['predict gain', 'msr'], ascending=[False, True])
    final_results = final_results[['code', 'name', 'timestamp', 'model', 'price', 'term', 'predict', 'predict gain', 'msr', 'ref val']]
    '''

    view_len = len(final_results)
    if view_len > 10:
        view_len = 10
    logger.info(final_results.head(view_len))
    print(final_results.head(view_len))
    # 予想収益が多い順に、100株ずつ買っていく戦略
    print('Recommended purchase method:')
    balance = int(args.now)
    idx = 0
    while balance > 0 and idx < len(final_results):
        look = final_results.iloc[idx]
        price = look['price'] * 100
        if price > balance:
            break
        idx += 1
        print(idx, look['code'], look['name'], 'Expected revenue:', (look['predict gain'] * 100))
        balance -= price
    time_end = time.perf_counter()
    elapsed = time_end - time_begin
    logger.info('estimate complete in ' + str(elapsed) + 's')
    print('done')
    # 完了通知
    notification.notify(
        title="complete estimating",
        message="complete estimating",
        app_name="estimate.py",
        timeout=10
    )


if __name__ == "__main__":
    main()