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

# 自作ロガー追加
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../logger'))
from logger import Logger
logger = Logger(__name__, 'model_3.log')

# データを0-1に正規化するためのツール
scaler = MinMaxScaler(feature_range=(0, 1))

# ファイル名と同じクラスを持ち、共通のestimate関数を用意する
class model_3:
    """Fit the model.

        LSTMによる推定(TODO)

        引数(TODO)
        ----------
        X : iterable
            Training data. Must fulfill input requirements of first step of the
            pipeline.

        y : iterable, default=None
            Training targets. Must fulfill label requirements for all steps of
            the pipeline.

        **fit_params : dict of string -> object
            Parameters passed to the ``fit`` method of each step, where
            each parameter name is prefixed such that parameter ``p`` for step
            ``s`` has key ``s__p``.

        戻り値(TODO)
        -------
        self : object
            Pipeline with fitted steps.
    """
    def __init__(self, code: str, filename: str):
        self.name = 'model_LSTM'
        self.code = code
        self.filename = filename

    def first_compile(self, stock_df: pl.DataFrame):
        closes = stock_df['Close'].to_numpy().reshape(-1, 1)
        # データを0~1の範囲に正規化
        scaled_closes = scaler.fit_transform(closes)
        # 全体の80%をトレーニングデータとして扱う
        training_data_len = int(np.ceil(len(closes) * .8))
        # どれくらいの期間をもとに予測するか
        window_size = 128
        # データ数が足りなければ終了
        # TODO:ここで終了するとmodelが作成されず、compile(),predict()呼び出し時にエラーになる
        if training_data_len <= window_size:
            self.msr = 10000
            logger.info('[' + str(self.code) + ']cannot learn because few data')
            return False

        train_data = scaled_closes[0:int(training_data_len), :]

        # train_dataをx_trainとy_trainに分ける
        x_train, y_train = [], []
        for i in range(window_size, len(train_data)):
            x_train.append(train_data[i - window_size:i, 0])
            y_train.append(train_data[i, 0])

        # numpy arrayに変換
        x_train, y_train = np.array(x_train), np.array(y_train)
        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

        model = Sequential()
        model.add(LSTM(units=50,return_sequences=True,input_shape=(x_train.shape[1], 1)))
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
        # TensorFlowのログを出力
        # 保存先ディレクトリがない場合は作成
        #import datetime
        #from pathlib import Path
        #dir_name = os.path.join(os.path.dirname(__file__), '../../../log/fit/', str(datetime.datetime.now().strftime("%Y%m%d-%H%M%S")))
        #dir = Path(dir_name)
        #dir.mkdir(parents=True, exist_ok=True)
        #profile_start_step = int(x_train.shape[0] * 1.5)
        #tensorboard_callback = tf.keras.callbacks.TensorBoard(
        #    log_dir=dir_name,
        #    histogram_freq=1,
        #    profile_batch='100, 120')
        #history = model.fit(x_train, y_train, batch_size=32, epochs=100, verbose=0, callbacks=[tensorboard_callback])
        history = model.fit(x_train, y_train, batch_size=32, epochs=100)
        
        # テストデータ(残り20%)作成
        test_data = scaled_closes[training_data_len - window_size:, :]
        #test_data = scaled_Y

        x_test = []
        y_test = closes[training_data_len:, :]
        for i in range(window_size, len(test_data)):
            x_test.append(test_data[i-window_size:i, 0])

        # numpy arrayに変換
        x_test = np.array(x_test)
        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

        #logger.info('x_test.shape:' + str(x_test.shape))
        #logger.info('y_test.shape:' + str(y_test.shape))
        
        predictions = model.predict(x_test)
        predictions = scaler.inverse_transform(predictions)
        #logger.info('predictions.shape:' + str(predictions.shape))

        # RMSE(二乗平均平方根誤差、0に近いほど良い)
        rmse = np.sqrt(np.mean((predictions - y_test) ** 2))
        print(f'RMSE:{rmse}')

        # モデル保存
        #model.save(self.filename)

        # 可視化
        real = stock_df.select(['Date', 'Close'])
        real = real.with_columns(pl.lit('Real').alias('Name'))
        predict_date = stock_df[training_data_len:].get_column('Date')
        predict = pl.from_numpy(predictions, schema=['Close'])
        predict = predict.with_columns(predict_date, pl.lit('Predict').alias('Name'))
        predict = predict.with_columns(pl.col('Close').cast(pl.Float64))
        df = pl.concat([real, predict.select(['Date', 'Close', 'Name'])])
        fig = px.line(x=df['Date'], y=df['Close'], color=df['Name'])
        fig.show()

        return True

'''
    def compile(self, delta_X, delta_Y, last_date: pd.Timestamp, *discard):
        logger.info('compile [' + str(self.code) + ']' + str(delta_X))
        # 日数を結合
        self.days = pd.concat([self.days, delta_X])
        # どれくらいの期間をもとに予測するか
        window_size = 60
        # データ数が足りなければ終了
        if len(self.scaled_Y) <= window_size:
            self.msr = 10000
            logger.info('[' + str(self.code) + ']cannot learn because few data')
            return
        training_data_begin = len(self.scaled_Y) - window_size
        Y = self.scaler.inverse_transform(self.scaled_Y)
        Y = np.concatenate([Y, delta_Y.to_numpy().reshape(-1, 1)])
        # データを0-1に正規化
        self.scaled_Y = self.scaler.fit_transform(Y)

        # モデル読み込み
        try:
            model = tf.keras.models.load_model(self.modelfile)
        except IOError:
            print('model is none')
            self.first_compile(Y)
            return

        # 渡された差分の100%をトレーニングデータとして扱う
        training_data_len = int(np.ceil(len(Y) * 1.0))
        #training_data_len = int(np.ceil(len(Y) * 1.0))

        train_data = self.scaled_Y[training_data_begin:int(training_data_len), :]

        # train_dataをx_trainとy_trainに分ける
        x_train = []
        y_train = train_data[window_size:, :]
        for i in range(window_size, len(train_data)):
            x_train.append(train_data[i - window_size:i, 0])

        # numpy arrayに変換
        x_train, y_train = np.array(x_train), np.array(y_train)
        logger.info('[' + str(self.code) + ']x_train.shape:' + str(x_train.shape) + ' y_train.shape:' + str(y_train.shape))
        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

        history = model.fit(x_train, y_train, batch_size=32, epochs=100, verbose=0)
        
        # モデル保存
        model.save(self.modelfile)
'''
        
'''
        # テストデータ(残り20%)作成
        test_data = self.scaled_Y[training_data_len - window_size:, :]
        #test_data = scaled_Y

        x_test = []
        y_test = Y[training_data_len:, :]
        for i in range(window_size, len(test_data)):
            x_test.append(test_data[i-window_size:i, 0])

        # numpy arrayに変換
        x_test = np.array(x_test)
        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

        logger.info('x_test.shape:' + str(x_test.shape))
        logger.info('y_test.shape:' + str(y_test.shape))
        
        predictions = self.model.predict(x_test)
        predictions = self.scaler.inverse_transform(predictions)
        logger.info('predictions.shape:' + str(predictions.shape))
'''

'''
        # MSR(平均二乗差)
        # self.msr = np.mean((predictions - y_test) ** 2)
        self.last_date = last_date
        
    def predict(self, days):
        window_size = 60
        x_test = []
        test_data = self.scaled_Y
        # 実データから作成(+1日まで)
        for i in range(window_size, len(test_data) + 1):
            x_test.append(test_data[i-window_size:i, 0])
        x_test = np.array(x_test)
        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
        # モデル読み込み
        model = tf.keras.models.load_model(self.modelfile)
        predictions = model.predict(x_test)
        # モデルの予想値を含めて作成
        for i in range(len(test_data) + 1, len(test_data) + days):
            test_data = np.append(test_data, predictions[-1])
            test_data = np.reshape(test_data, (test_data.shape[0], 1))
            x_test_one = []
            x_test_one.append(test_data[i-window_size:i, 0])
            x_test_one = np.array(x_test_one)
            x_test_one = np.reshape(x_test_one, (x_test_one.shape[0], x_test_one.shape[1], 1))
            predictions = np.append(predictions, model.predict(x_test_one))
            predictions = np.reshape(predictions, (predictions.shape[0], 1))
        logger.info('predictions.shape:' + str(predictions.shape))
        # 正規化を元に戻す
        y_hat = self.scaler.inverse_transform(predictions)
        # 1次元化
        y_hat = y_hat.ravel()
        first_day = self.days.iloc[0] + window_size
        last_day = self.days.iloc[-1] + window_size + days
        ret_days = np.arange(first_day, last_day + 1, 1)
        return (ret_days, y_hat)
'''

def set_argparse():
    parser = argparse.ArgumentParser(description='LSTMで予想を出す')
    parser.add_argument('input', help='株価データが保存されたファイルまたはディレクトリ')
    parser.add_argument('-o', '--output', help='予想の出力ファイル。ただし、inputの対象が1ファイルのときのみ有効', default='')
    args = parser.parse_args()
    return args

def main():
    args = set_argparse()
    data_files = []
    '''
    if os.path.isdir(args.input):
        data_files = glob.glob(args.input + '/*.parquet')
    else:
        data_files = [args.input]
    '''
    data_files = [args.input]

    for index in tqdm.tqdm(range(len(data_files))):
        data_file = data_files[index]
        # 株価データ読み込み
        try:
            stock_df = pl.read_parquet(data_file)
        except:
            print('株価データファイルの読み込みに失敗しました')
            sys.exit(1)
        '''
        # 出力先ファイル決定
        out_strategy_file = args.output
        if out_strategy_file == '' or len(data_files) > 1:
            # 保存先ディレクトリがない場合は作成
            dir = Path(out_estimate_directory_default)
            dir.mkdir(parents=True, exist_ok=True)
            out_strategy_file = os.path.join(
                out_estimate_directory_default, os.path.splitext(os.path.basename(data_file))[0]
            ) + '.json'
        '''
        code = stock_df.get_column('Code')[0]
        model = model_3(code, f'model_lstm_{code}')
        model.first_compile(stock_df)

        #estimate(stock_df, out_strategy_file)

    sys.exit(0)

if __name__ == "__main__":
    main()
