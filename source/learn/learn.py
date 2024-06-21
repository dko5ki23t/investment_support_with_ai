import model            # 学習モデル
import model_global     # 学習モデル
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
    time_begin = time.perf_counter()

    #############################################
    # 全銘柄をinputとする学習
    #############################################
    # TODO:名前、globalというかacross??

    # 全銘柄を含めたDataFrameを作成する
    # date(日付)を元に結合、最後にdateはindexにする
    # その銘柄のデータに対応する日付のcloseがない場合はNaNが入る
    df_closes_global = pd.DataFrame(columns=['date', 'close'])
    # ファイル読み込み
    files = glob.glob(str(args.dir) + '/*.pkl')
    # TODO:(一旦)日経平均株価は除外
    files = [e for e in files if not e.endswith('N225.pkl')]
    for file in files:
        df = pd.read_pickle(file)
        code = str(df['code'].iloc[0])
        df = df[['date', 'close']]
        df = df.rename(columns={'close': code})
        df_closes_global = pd.merge(df_closes_global, df, on='date', how='outer', sort=True)
    df_closes_global = df_closes_global.set_index('date')
    df_closes_global = df_closes_global.drop('close', axis=1)
    # TODO: 日経平均株価を除外した結果、NaNのみの行ができる可能性あり→削除している
    df_closes_global = df_closes_global.dropna(how='all')
    # TODO:何も考えずNaN→0に置換している。機械学習的に精度悪くなりそう・・・
    df_closes_global = df_closes_global.fillna(0.0)

    # 差分だけ学習をするスタイルはとらないことにした。
    # （後続の、各銘柄をinputとする方はこれを決定する前に実装したものが残っている）
    # 理由は、inputデータを正規化しているから。差分と、元からある正規化済みデータとの整合を取るのが難しく、現在それを実現できていない。
    # 差分を学習しようと思ったのは、時間短縮のためだったが、全銘柄inputだとそこまで時間がかからない。
    # 差分を学習する以外の方法で時間短縮を図った方が良さそう？

    models_file_name = str(args.out_dir) + '/global.pkl'
    try:
        models = model_global.estimate_global(df_closes_global, str(args.out_dir))
    except Exception as e:
        logger.error(e)
    # 学習結果の保存
    f = open(models_file_name, 'wb')
    pickle.dump(models, f)
    f.close


    #############################################
    # 各銘柄をinputとする学習
    #############################################
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
            for m in models:
                # last_dateをもとに、差分を渡す
                # ※last_dateはfetched_dataの最新の時刻が同期されている
                # TODO: 現在は、日付が同じで時刻が遅くても学習対象にはしていない
                #       (学習を一つ取り消す方法を見つけるか、時刻情報含めて学習すれば同じ日付も学習対象としていいと思う)
                df_delta = df[df['date'] > m.last_date.date()]
                logger.info(str(df['code'].iloc[0]) + ' ' + str(m.name) + ' delta days:' + str(len(df_delta)))
                if len(df_delta) > 0:
                    # TODO: compileに渡す引数を統一
                    if m.name == 'model5':
                        m.compile(df_delta['day'], df_delta[['close', 'high', 'volume']].to_numpy(copy=True), df['timestamp'].iloc[-1])
                    else:
                        m.compile(df_delta['day'], df_delta['close'], df['timestamp'].iloc[-1])
                # 上のcompile()呼び出し時にlast_dateは更新されている
            # meta_dataのlast_dateも更新（TODO: meta_data不要？）
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
    print('done')
    # 完了通知
    notification.notify(
        title="complete learning",
        message="complete learning",
        app_name="learn.py",
        timeout=10
    )

if __name__ == "__main__":
    main()