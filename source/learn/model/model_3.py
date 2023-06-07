import pandas as pd
import numpy as np
import os
#os.environ['TF_CPP_MIN_LOG_LEVEL']='2'   # TensorFlowの警告を出力しない
os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private' # GPU占有化
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
import tensorflow as tf
#tf.debugging.set_log_device_placement(True)

# 自作ロガー追加
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../../logger'))
from logger import Logger
logger = Logger(__name__, 'model_3.log')

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
    def __init__(self, code, stock, X, Y, last_date: pd.Timestamp, dir: str, *discard):
        self.name = 'model3'
        self.code = code
        self.stock = stock
        self.days = X
        self.last_date = last_date
        self.modelfile = dir + '/models/' + str(code) + '_' + self.name

        # データを0-1に正規化
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.scaled_Y = self.scaler.fit_transform(Y)
        self.first_compile(Y)

    def first_compile(self, Y):
        # 全体の80%をトレーニングデータとして扱う
        training_data_len = int(np.ceil(len(Y) * .8))
        # どれくらいの期間をもとに予測するか
        window_size = 60
        # データ数が足りなければ終了
        # TODO:ここで終了するとmodelが作成されず、compile(),predict()呼び出し時にエラーになる
        if training_data_len <= window_size:
            self.msr = 10000
            logger.info('[' + str(self.code) + ']cannot learn because few data')
            return False

        train_data = self.scaled_Y[0:int(training_data_len), :]

        # train_dataをx_trainとy_trainに分ける
        x_train = []
        y_train = train_data[window_size:, :]
        for i in range(window_size, len(train_data)):
            x_train.append(train_data[i - window_size:i, 0])

        # numpy arrayに変換
        x_train, y_train = np.array(x_train), np.array(y_train)
        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

        model = Sequential()
        model.add(LSTM(units=50,return_sequences=True,input_shape=(x_train.shape[1], 1)))
        model.add(Dropout(0.2))
        model.add(LSTM(units=50,return_sequences=True))
        model.add(Dropout(0.2))
        model.add(LSTM(units=50,return_sequences=True))
        model.add(Dropout(0.2))
        model.add(LSTM(units=50))
        model.add(Dropout(0.2))
        model.add(Dense(units=1))

        print('compile start (for the first time)')
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
        history = model.fit(x_train, y_train, batch_size=32, epochs=100, verbose=0)
        
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

        #logger.info('x_test.shape:' + str(x_test.shape))
        #logger.info('y_test.shape:' + str(y_test.shape))
        
        predictions = model.predict(x_test)
        predictions = self.scaler.inverse_transform(predictions)
        #logger.info('predictions.shape:' + str(predictions.shape))

        # MSR(平均二乗差)
        self.msr = np.mean((predictions - y_test) ** 2)

        # モデル保存
        model.save(self.modelfile)
        return True

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
        model = tf.keras.models.load_model(self.modelfile)
        if (model is None):
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