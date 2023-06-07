import model            # 学習モデル
import argparse         # コマンドライン引数チェック用
import pandas as pd
import glob
from pathlib import Path
import tqdm
import time
import pickle
from plyer import notification

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
    # 日経平均株価は除外(別用途で使う)
    files = [e for e in files if not e.endswith('N225.pkl')]
    try:
        nikkei_df = pd.read_pickle(args.dir + '/N225.pkl')
    except Exception as e:
        print('cannot open Nikkei average file')
        print('this may cause that some model cannot learn')
        nikkei_df = None
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
            f = open(models_file_name, 'rb')
            models = pickle.load(f)
            f.close
            meta_data = models.pop(0)
            # last_dateをもとに、差分を渡す
            # ※last_dateはfetched_dataの最新の時刻が同期されている
            # TODO: 現在は、日付が同じで時刻が遅くても学習対象にはしていない
            #       (学習を一つ取り消す方法を見つけるか、時刻情報含めて学習すれば同じ日付も学習対象としていいと思う)
            df_delta = df[df['date'] > meta_data['last_date'].date()]
            logger.info(str(df['code'].iloc[0]) + ' delta days:' + str(len(df_delta)))
            if len(df_delta) > 0:
                for m in models:
                    # TODO: compileに渡す引数を統一
                    if m.name == 'model5':
                        m.compile(df_delta['day'], df_delta[['close', 'high', 'volume']].to_numpy(copy=True), df['timestamp'].iloc[-1])
                    else:
                        m.compile(df_delta['day'], df_delta['close'], df['timestamp'].iloc[-1])
            # last_date更新
            meta_data['last_date'] = df['timestamp'].iloc[-1]
            models.insert(0, meta_data)
        else:
            try:
                models = model.estimate(df, nikkei_df, str(args.out_dir))
            except Exception as e:
                logger.error(e)
                continue
        # 学習結果の保存
        f = open(models_file_name, 'wb')
        pickle.dump(models, f)
        f.close
    time_end = time.perf_counter()
    elapsed = time_end - time_begin
    logger.info('learn complete in ' + str(elapsed) + 's')
    # 完了通知
    notification.notify(
        title="complete learning",
        message="complete learning",
        app_name="learn.py",
        timeout=10
    )

if __name__ == "__main__":
    main()