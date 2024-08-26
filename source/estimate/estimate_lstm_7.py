import argparse         # コマンドライン引数チェック用
import numpy as np
import polars as pl
import os
import tqdm
#os.environ['TF_CPP_MIN_LOG_LEVEL']='2'   # TensorFlowの警告を出力しない
os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private' # GPU占有化
#from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
import tensorflow as tf
#import matplotlib.pyplot as plt
import plotly.express as px
#tf.debugging.set_log_device_placement(True)
from pathlib import Path
import json
import glob
import datetime
import jpholiday

# 自作ロガー追加
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../logger'))
from logger import Logger
logger = Logger(__name__, 'model_lstm_7.log')

# 機械学習の乱数シードを固定
tf.random.set_seed(1234)

input_directory_default = os.path.join(os.path.dirname(__file__), '../../db/stock_data')
stock_info_file_default = os.path.join(os.path.dirname(__file__), '../../db/stock_info.csv')
out_estimate_directory_default = os.path.join(os.path.dirname(__file__), '../../db/estimates/estimate_lstm_7')
model_file_default = os.path.join(os.path.dirname(__file__), '../../model/model_lstm_7')

# データを0-1に正規化するためのツール
scaler = MinMaxScaler(feature_range=(0, 1))
scaler2 = MinMaxScaler(feature_range=(0, 1))
scaler3 = MinMaxScaler(feature_range=(0, 1))
scaler4 = MinMaxScaler(feature_range=(0, 1))

def name():
    """
    推定方法の名前
    """
    return 'LSTM7'

def version():
    """
    バージョン
    """
    return '1.0'

def description():
    """
    説明
    """
    return 'LSTMによる推定を行う。一定期間(window_size日数分)の終値・出来高と日経平均終値をもとに、次の日の（高値 - 始値）を推定する。与えられたデータの後半20%を用いて推定値を出し、評価値も出す。'

# 次の営業日を返す
def next_business_day(date: str):
    cur_date = datetime.datetime.strptime(date, "%Y-%m-%d")
    while True:
        cur_date = cur_date + datetime.timedelta(days=1)
        if cur_date.weekday() < 5 and not jpholiday.is_holiday(cur_date):
            return datetime.datetime.strftime(cur_date, "%Y-%m-%d")

def estimate(n225_df: pl.DataFrame, stock_df: pl.DataFrame, out_file: str, model_file='', window_size=128, show_figure=False, *discard):
    """
    LSTMによる推定を行う。
    一定期間(window_size日数分)の終値・出来高と日経平均終値をもとに、次の日の（高値 - 始値）を推定する。
    与えられたデータの後半20%を用いて推定値を出し、評価値も出す。

    n225_df,
    stock_df : DataFrame
                推定のもととなる株価データ
                必要な列 : 'Date', 'Code', 'Open', 'Close', 'Low', 'High', 'Volume'

    out_file : str
                推定結果の出力先ファイル名

    model_file : str, default=''
                LSTMのモデル（fit済）が保存されたファイル名。存在する場合は読み込んで推定値を出す。
                空文字列を指定した場合は新規にモデルを構築する。（時間がかかる）

    window_size : int, default=128
                何日間の終値を元に次の日の終値を推定するか

    show_figure : bool, default=False
                実際のデータと推定値の比較用グラフを出力するか
    """

    code = stock_df.get_column('Code')[0]
    stock_df_with_n225 = stock_df.join(
        n225_df.select(pl.col('Date'), pl.col('Close').alias('N225_Close')),
        on=['Date'],
        how="left")
    # 日経平均にはデータがない日もある(null値)ため、その行は削除
    stock_df_with_n225 = stock_df_with_n225.drop_nulls()
    closes = stock_df_with_n225['Close'].to_numpy().reshape(-1, 1)
    volumes = stock_df_with_n225['Volume'].to_numpy().reshape(-1, 1)
    n225_closes = stock_df_with_n225['N225_Close'].to_numpy().reshape(-1, 1)
    # 高値-始値の差分
    stock_df_with_delta = stock_df_with_n225.with_columns((pl.col('High') - pl.col('Open')).alias('Delta'))
    deltas = stock_df_with_delta['Delta'].to_numpy().reshape(-1, 1)
    # データを0~1の範囲に正規化
    scaled_values = np.stack([
        scaler.fit_transform(closes).flatten(),
        scaler3.fit_transform(volumes).flatten(),
        scaler4.fit_transform(n225_closes).flatten()], axis=1)
    scaled_deltas = scaler2.fit_transform(deltas)
    # モデル読み込み
    if model_file != '':
        try:
            model = tf.keras.models.load_model(model_file)
        except IOError:
            logger.info('モデルを読み込めませんでした。新たにモデルを構築します。')
            model = build_model(scaled_values, scaled_deltas, code, model_file, window_size)
    else:
        model = build_model(scaled_values, scaled_deltas, code, model_file, window_size)
    # モデル作成に失敗した場合はそのままreturn
    if model is None:
        logger.error('モデル構築に失敗しました。')
        return

    # テストデータ(残り20%)作成
    training_data_len = int(np.ceil(len(scaled_values) * .8))
    # データ数が足りなければ終了
    if training_data_len <= window_size:
        logger.info(f'[{code}]データ数が足りないため評価できませんでした。')
        return
    test_data_x = scaled_values[training_data_len - window_size:, :]

    x_test = []
    y_test = deltas[training_data_len:, :]
    scaled_y_test = scaled_deltas[training_data_len:, :]
    for i in range(window_size, len(test_data_x)):
        x_test.append(test_data_x[i-window_size:i, :])

    # numpy arrayに変換
    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 3))

    scaled_predictions = model.predict(x_test)
    predictions = scaler2.inverse_transform(scaled_predictions)
    
    # RMSE(二乗平均平方根誤差、0に近いほど良い)
    rmse = np.sqrt(np.mean((predictions - y_test) ** 2))
    logger.info(f'[{code}]RMSE:{rmse}')
    rmse2 = np.sqrt(np.mean((scaled_predictions - scaled_y_test) ** 2))
    logger.info(f'[{code}]RMSE2:{rmse2}')

    # 評価用ではなく、与えられたデータにない次の日の推測値を出す
    x_predict = [test_data_x[len(test_data_x)-window_size:len(test_data_x), :]]
    # numpy arrayに変換
    x_predict = np.array(x_predict)
    x_predict = np.reshape(x_predict, (x_predict.shape[0], x_predict.shape[1], 3))
    prediction_newday = model.predict(x_predict)
    prediction_newday = scaler2.inverse_transform(prediction_newday)
    predictions = np.append(predictions, prediction_newday, axis=0)

    real = stock_df_with_delta.select(['Date', 'Delta'])
    real = real.with_columns(pl.lit('Real').alias('Name'))
    predict_date = stock_df_with_delta[training_data_len:].get_column('Date')
    # 最後の次の日も追加
    predict_date.append(pl.Series('Date', [next_business_day(predict_date[-1])]))
    predict = pl.from_numpy(predictions, schema=['Delta'])
    predict = predict.with_columns(predict_date, pl.lit('Predict').alias('Name'))
    predict = predict.with_columns(pl.col('Delta').cast(pl.Float64))
    predict = predict.select(['Date', 'Delta', 'Name'])
    # 可視化
    if show_figure:
        df = pl.concat([real, predict])
        fig = px.line(x=df['Date'], y=df['Delta'], color=df['Name'])
        fig.show()

    # 予想を出力        
    out_list = []
    prev_close = stock_df_with_n225.filter(pl.col('Date') == predict.get_column('Date')[0]).get_column('Close')[0]
    # TODO: iter_rows()は非推奨
    i = 0
    for row in predict.iter_rows():
        if row[1] > 0:
            # 過去20日分のRMSE(二乗平均平方根誤差、0に近いほど良い)
            rmse3 = rmse
            rmse4 = rmse2
            rmse5 = rmse  # 5日
            rmse6 = rmse2
            rmse7 = rmse  # 10日
            rmse8 = rmse2
            rmse9 = rmse  # 50日
            rmse10 = rmse2
            rmse11 = rmse # 100日
            rmse12 = rmse2
            if i > 0:
                s = max(i-20, 0)
                rmse3 = np.sqrt(np.mean((predictions[s:i] - y_test[s:i]) ** 2))
                rmse4 = np.sqrt(np.mean((scaled_predictions[s:i] - scaled_y_test[s:i]) ** 2))
                s = max(i-5, 0)
                rmse5 = np.sqrt(np.mean((predictions[s:i] - y_test[s:i]) ** 2))
                rmse6 = np.sqrt(np.mean((scaled_predictions[s:i] - scaled_y_test[s:i]) ** 2))
                s = max(i-10, 0)
                rmse7 = np.sqrt(np.mean((predictions[s:i] - y_test[s:i]) ** 2))
                rmse8 = np.sqrt(np.mean((scaled_predictions[s:i] - scaled_y_test[s:i]) ** 2))
                s = max(i-50, 0)
                rmse9 = np.sqrt(np.mean((predictions[s:i] - y_test[s:i]) ** 2))
                rmse10 = np.sqrt(np.mean((scaled_predictions[s:i] - scaled_y_test[s:i]) ** 2))
                s = max(i-100, 0)
                rmse11 = np.sqrt(np.mean((predictions[s:i] - y_test[s:i]) ** 2))
                rmse12 = np.sqrt(np.mean((scaled_predictions[s:i] - scaled_y_test[s:i]) ** 2))
            out_list.append({
                "date": row[0], "code": code, "gains": row[1],
                "score": -rmse, "score2": -rmse2, "score3": -rmse3,
                "score4": -rmse4, "score5": -rmse5, "score6": -rmse6,
                "score7": -rmse7, "score8": -rmse8, "score9": -rmse9,
                "score10": -rmse10, "score11": -rmse11, "score12": -rmse12,
                "yest_close": prev_close
            })
        tmp = stock_df_with_n225.filter(pl.col('Date') == row[0])
        if len(tmp) > 0:
            prev_close = tmp.get_column('Close')[0]
        i = i + 1
    output = {
        "code": code,
        "method_name": name(),
        "version": version(),
        "last_date": stock_df_with_n225.get_column('Date')[-1],
        "estimate": out_list,
    }
    # ファイル出力
    with open(out_file, 'w') as f:
        json.dump(output, f, indent=2)

def build_model(scaled_values: np.ndarray, scaled_deltas: np.ndarray, code: str, model_file='', window_size=128):
    # 全体の80%をトレーニングデータとして扱う
    training_data_len = int(np.ceil(len(scaled_values) * .8))
    # データ数が足りなければ終了
    if training_data_len <= window_size:
        logger.info(f'[{code}]データ数が足りないため学習できませんでした。')
        return None

    train_data_x = scaled_values[0:int(training_data_len), :]
    train_data_y = scaled_deltas[0:int(training_data_len), :]

    # train_dataをx_trainとy_trainに分ける
    x_train, y_train = [], []
    for i in range(window_size, len(train_data_x)):
        x_train.append(train_data_x[i - window_size:i, :])
        y_train.append(train_data_y[i, :])

    # numpy arrayに変換
    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 3))

    model = Sequential()
    model.add(LSTM(units=50,return_sequences=True,input_shape=(x_train.shape[1], 3)))
    model.add(Dropout(0.25))
    model.add(LSTM(units=50,return_sequences=True))
    model.add(Dropout(0.25))
    model.add(LSTM(units=50,return_sequences=True))
    model.add(Dropout(0.25))
    model.add(LSTM(units=50))
    model.add(Dropout(0.25))
    model.add(Dense(units=1))

    print('コンパイル開始（初回）')
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
    history = model.fit(x_train, y_train, batch_size=32, epochs=50)
    # モデル保存
    filename = model_file
    if filename == '':
        filename = model_file_default + '_' + code
    model.save(filename)
    return model

def estimate_gen(input='', output='', stock_info_file='', filter_market_code=0, force_build_model=False, show_figure=False):
    """
    【ジェネレータ】LSTMによる推定を行う

    input : str, default=''
            株価データが保存されたファイルまたはディレクトリ

    output : str, default=''
            推定結果の出力ファイル。ただし、inputの対象が1ファイルのときのみ有効

    stock_info_file : str, default=''
            全銘柄の情報（銘柄名や市場コード等）が記載されたCSVファイル

    filter_market_code : int, default=0
            市場コードによるフィルタ。0の場合はフィルタリングしない

    force_build_model : bool, default=False
            必ず新規でモデルを構築するかどうか

    show_figure : bool, default=False
            実際のデータと推定値の比較用グラフを出力するか

    戻り値 : list
            index0 : 処理が終了したインデックス。 index1 : 処理総数
    """
    input_internal = input
    if input_internal == '':
        input_internal = input_directory_default
    # 銘柄情報ファイル読み込み
    stock_info_file_internal = stock_info_file
    if stock_info_file_internal == '':
        stock_info_file_internal = stock_info_file_default
    try:
        stock_info_df = pl.read_csv(stock_info_file_internal)
    except:
        print('銘柄情報データファイルの読み込みに失敗しました')
        sys.exit(1)
    if os.path.isdir(input_internal):
        data_files = glob.glob(input_internal + '/*.parquet')
    else:
        data_files = [input_internal]

    # 市場コードでフィルタリング
    filtered_stock_dfs = []
    # 日経平均のデータフレーム作成
    try:
        n225_df = pl.read_parquet(input_directory_default + '/N225/N225.parquet')
    except:
        print('日経平均株価データファイルの読み込みに失敗しました')
        sys.exit(1)
    for data_file in data_files:
        # 株価データ読み込み
        try:
            stock_df = pl.read_parquet(data_file)
        except:
            print('株価データファイルの読み込みに失敗しました')
            sys.exit(1)
        code = stock_df.get_column('Code')[0]
        if code[0] == '^':
            continue
        market_code = stock_info_df.filter(pl.col('Code') == code).get_column('MarketCode')[0]
        if filter_market_code == 0 or market_code == filter_market_code:
            filtered_stock_dfs.append(stock_df)

    for index in tqdm.tqdm(range(len(filtered_stock_dfs))):
        stock_df = filtered_stock_dfs[index]
        code = stock_df.get_column('Code')[0]
        # 出力先ファイル決定
        out_estimate_file = output
        if out_estimate_file == '' or len(data_files) > 1:
            # 保存先ディレクトリがない場合は作成
            dir = Path(out_estimate_directory_default)
            dir.mkdir(parents=True, exist_ok=True)
            out_estimate_file = os.path.join(
                out_estimate_directory_default, code
            ) + '.json'

        model_file = model_file_default + '_' + code
        if force_build_model:
            model_file = ''
        estimate(n225_df, stock_df, out_estimate_file, model_file=model_file, show_figure=show_figure)
        yield [index, len(filtered_stock_dfs)]

def set_argparse():
    parser = argparse.ArgumentParser(description='LSTMで予想を出す')
    parser.add_argument('input', help='株価データが保存されたファイルまたはディレクトリ')
    parser.add_argument('-m', '--filter_market_code', help='inputで指定したディレクトリ内の株価データを市場コードでフィルタリング', type=int, default=0)
    parser.add_argument('--stock_info', help='全銘柄の情報（銘柄名や市場コード等）が記載されたCSVファイル', default='')
    parser.add_argument('--force_build_model', help='必ず新規でモデルを構築する。このオプションを付けない場合はモデルが保存されたディレクトリがある場合はそのファイルからモデルを読み込む', action='store_true')
    parser.add_argument('--show_figure', help='実際のデータと推定値の比較用グラフを出力する。各銘柄ごとに出力されるため、inputにディレクトリを指定している場合は注意', action='store_true')
    parser.add_argument('-o', '--output', help='予想の出力ファイル。ただし、inputの対象が1ファイルのときのみ有効', default='')
    args = parser.parse_args()
    return args

def main():
    args = set_argparse()
    for i in estimate_gen(args.input, args.output, args.stock_info, args.filter_market_code, args.force_build_model, args.show_figure):
        pass

    sys.exit(0)

if __name__ == "__main__":
    main()
