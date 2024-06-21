import model            # 学習モデル
import argparse         # コマンドライン引数チェック用
import pandas as pd
import glob
from pathlib import Path
import tqdm
import time
import pickle
import sys

# 自作ロガー追加
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../logger'))
from logger import Logger
logger = Logger(__name__, 'learn_historical.log')

def set_argparse():
    parser = argparse.ArgumentParser(description='TODO')
    parser.add_argument('file', help='stock data file')
    parser.add_argument('idx', help='stock data idx')
    parser.add_argument('out_dir', help='学習データ保存先')
    args = parser.parse_args()
    return args

def main():
    args = set_argparse()
    # 保存先ディレクトリがない場合は作成
    dir = Path(args.out_dir)
    dir.mkdir(parents=True, exist_ok=True)
    time_begin = time.perf_counter()

    file = args.file
    df = pd.read_pickle(file)
    # 引数指定のインデックスまでを抽出
    if int(args.idx) > len(df):
        sys.exit(5)     # インデックスが大きすぎることを通知
    df = df[:int(args.idx)]
    models_file_name = str(args.out_dir) + '/' + str(df['code'].iloc[0]) + '.pkl'
    # 今は使ってない (TODO)
    nikkei_df = pd.DataFrame()
    # 既に学習モデルファイルが存在するか確認
    models = None
    if os.path.exists(models_file_name):     # 差分のみ学習
        f = open(models_file_name, 'rb')
        models = pickle.load(f)
        f.close
        meta_data = models.pop(0)
        # last_dateをもとに、差分を渡す
        df_delta = df[df['timestamp'] > meta_data['last_date']]
        logger.info(str(df['code'].iloc[0]) + ' delta days:' + str(len(df_delta)))
        if len(df_delta) > 0:
            for m in models:
                # TODO: compileに渡す引数を統一
                if m.name == 'model5':
                    m.compile(df_delta['day'], df_delta[['close', 'high', 'volume']].to_numpy(copy=True), df['timestamp'].iloc[-1])
                else:
                    m.compile(df_delta['day'], df_delta['close'], df['timestamp'].iloc[-1])
        models.insert(0, meta_data)
    else:
        logger.info('[' + str(df['code'].iloc[0]) + ']')
        logger.info(df)
        try:
            models = model.estimate(df, nikkei_df)
        except Exception as e:
            print(e)
    # 学習結果の保存
    f = open(models_file_name, 'wb')
    pickle.dump(models, f)
#        models.to_pickle(models_file_name)
    f.close
    time_end = time.perf_counter()
    elapsed = time_end - time_begin
    logger.info('learn complete in ' + str(elapsed) + 's')

if __name__ == "__main__":
    main()