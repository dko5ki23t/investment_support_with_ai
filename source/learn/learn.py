import model            # 学習モデル
import argparse         # コマンドライン引数チェック用
import pandas as pd
import glob
from pathlib import Path
import tqdm
import time

# 自作ロガー追加
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../logger'))
from logger import Logger
logger = Logger(__name__, 'learn.log')

def set_argparse():
    parser = argparse.ArgumentParser(description='TODO')
    parser.add_argument('dir', help='stock data directory')
    parser.add_argument('out_dir', help='学習データ保存先')
    args = parser.parse_args()
    return args

def main():
    args = set_argparse()
    # ファイル読み込み
    files = glob.glob(args.dir + '/*.pkl')
    # 保存先ディレクトリがない場合は作成
    dir = Path(args.out_dir)
    dir.mkdir(parents=True, exist_ok=True)
    time_begin = time.perf_counter()
    for index in tqdm.tqdm(range(len(files))):
        file = files[index]
        df = pd.read_pickle(file)
        models_file_name = str(args.out_dir) + '/' + str(df['code'].iloc[0]) + '.pkl'
        # 既に学習モデルファイルが存在するか確認
        models = None
        if os.path.exists(models_file_name):     # 差分のみ学習
            models = pd.read_pickle(models_file_name)
            # last_dateをもとに、差分を渡す
            df_delta = df[df['timestamp'] > models[0].last_date]
            logger.info(str(df['code'].iloc[0]) + ' delta days:' + str(len(df_delta)))
            if len(df_delta) > 0:
                for m in models:
                    m.compile(df_delta['day'], df_delta['close'], df['timestamp'].iloc[-1])
        else:
            logger.info('[' + str(df['code'].iloc[0]) + ']')
            logger.info(df)
            try:
                models = model.estimate(df, 'day', 'close')
            except Exception as e:
                logger.error(e)
                continue
        # 学習結果の保存
        models.to_pickle(models_file_name)
    time_end = time.perf_counter()
    elapsed = time_end - time_begin
    logger.info('learn complete in ' + str(elapsed) + 's')

if __name__ == "__main__":
    main()