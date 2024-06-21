import argparse         # コマンドライン引数チェック用
import pandas as pd
import glob
from pathlib import Path
import tqdm
import time
import pickle
import sys
import signal
import numpy as np

# 自作ロガー追加
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../logger'))
from logger import Logger
logger = Logger(__name__, 'history.log')
# 自作モデル追加
sys.path.append(os.path.join(os.path.dirname(__file__), '../learn'))
import model

# 学習モデル
models = None
# 学習モデル保存先
models_file_name = ''

def save_models(signum, frame):
    print('catch SIGINT')
    # 学習結果の保存
    f = open(models_file_name, 'wb')
    pickle.dump(models, f)
    f.close
    print('save models to file:' + models_file_name)
    sys.exit(0)

def set_argparse():
    parser = argparse.ArgumentParser(description='TODO')
    parser.add_argument('file', help='stock data file')
    parser.add_argument('idx', help='stock data idx')
    parser.add_argument('intermediate', help='中間ファイル')
    parser.add_argument('learn_out', help='学習データ保存先')
    parser.add_argument('estimate_out', help='推測データ保存先')
    parser.add_argument('term', help='何日後を予測するか')
    args = parser.parse_args()
    return args

def main():
    args = set_argparse()
    # 保存先ディレクトリがない場合は作成
    learn_dir = Path(args.learn_out)
    learn_dir.mkdir(parents=True, exist_ok=True)
    estimate_dir = Path(args.estimate_out)
    estimate_dir.mkdir(parents=True, exist_ok=True)
    # シグナルハンドラ登録
    signal.signal(signal.SIGINT, save_models)

    time_begin = time.perf_counter()

    file = args.file
    df = pd.read_pickle(file)
    fetched_idx = int(args.idx)
    max_idx = int(len(df))
    models_file_name = str(learn_dir) + '/' + str(df['code'].iloc[0]) + '.pkl'
    # 既に学習モデルファイルが存在するか確認
    if os.path.exists(models_file_name):     # ファイルから学習モデルを取得
        f = open(models_file_name, 'rb')
        models = pickle.load(f)
        f.close
    # 予測DataFrame
    predict_results = pd.DataFrame(columns=['code', 'name', 'timestamp', 'date',
                                            'model', 'price', 'term', 'predict',
                                            'predict gain', 'actual', 'msr'])

    while True:
        if fetched_idx > max_idx:
            break
        print('learn start (' + str(fetched_idx) + '/' + str(max_idx) + ')')
        # 引数指定のインデックスまでを抽出
        extracted_df = df[:fetched_idx]
        # 今は使ってない (TODO)
        nikkei_df = pd.DataFrame()
        
        if model is not None:     # 差分のみ学習
            meta_data = models.pop(0)
            # last_dateをもとに、差分を渡す
            df_delta = extracted_df[extracted_df['timestamp'] > meta_data['last_date']]
            if len(df_delta) > 0:
                for m in models:
                    # TODO: compileに渡す引数を統一
                    if m.name == 'model5':
                        m.compile(df_delta['day'], df_delta[['close', 'high', 'volume']].to_numpy(copy=True), extracted_df['timestamp'].iloc[-1])
                    else:
                        m.compile(df_delta['day'], df_delta['close'], extracted_df['timestamp'].iloc[-1])
            models.insert(0, meta_data)
        else:
            try:
                models = model.estimate(extracted_df, nikkei_df)
            except Exception as e:
                print(e)

        print('estimate start (' + str(fetched_idx) + '/' + str(max_idx) + ')')
        now_val = extracted_df['close'].iloc[-1]      # 最新の終値を現在の株価とする
        for model in models:
            try:
                (days, predict_vals) = model.predict(1)
                tmp_s = pd.Series(
                    [model.code, model.stock, extracted_df['timestamp'].iloc[-1], extracted_df['timestamp'].iloc[-1].date(),
                    model.name, now_val, int(args.term), predict_vals[-1],
                    (predict_vals[-1] - now_val), np.nan, model.msr],
                    index=predict_results.columns)
            except Exception as e:
                logger.error('[' + str(model.code) + '] ' + str(model.name) + ':')
                logger.error(e)
                fetched_idx = fetched_idx + 1
                continue
            predict_results = pd.concat([predict_results, pd.DataFrame(data=tmp_s.values.reshape(1, -1), columns=predict_results.columns)])
        
        fetched_idx = fetched_idx + 1

    time_end = time.perf_counter()
    elapsed = time_end - time_begin
    logger.info('complete in ' + str(elapsed) + 's')

if __name__ == "__main__":
    main()