import pandas as pd
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'   # TensorFlowの警告を出力しない
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
import tensorflow as tf

# 機械学習の乱数シードを固定
tf.random.set_seed(1234)

#tf.debugging.set_log_device_placement(True)

# 自作ロガー追加
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../../logger'))
from logger import Logger
logger = Logger(__name__, 'model_4.log')


# 予測結果を正規化する前にdatasetと同じnumpy形式に変換する
def padding_array(val, size):
    # 作業用のnumpy配列を用意する
    temp_column = np.zeros(size)
    xset = []
    for x in val:
        a = np.insert(temp_column, 0, x)
        xset.append(a)

    xset = np.array(xset)
    return xset

# ファイル名と同じクラスを持ち、共通のestimate関数を用意する
class model_4:
    """Fit the model.

        LSTMによる推定(TODO) 日経平均株価も説明変数に追加

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
    def __init__(self, code: int, stock: str, X: pd.Series, Y: np.ndarray,
                 last_date: pd.Timestamp, *discard):
        self.name = 'model4'
        self.code = code
        self.stock = stock
        self.days = X
        self.last_date = last_date
        self.model = None

        # データを0-1に正規化
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.scaled_Y = self.scaler.fit_transform(Y)
        self.first_compile(Y)

    def first_compile(self, Y: np.ndarray):
        # 全体の80%をトレーニングデータとして扱う
        training_data_len = int(np.ceil(len(Y) * .8))
        # どれくらいの期間をもとに予測するか
        window_size = 60
        # データ数が足りなければ終了
        if training_data_len <= window_size:
            self.msr = 10000
            logger.info('cannot learn [' + str(self.code) + '] because few data')
            return False

        train_data = self.scaled_Y[0:int(training_data_len), :]

        # train_dataをx_trainとy_trainに分ける
        x_train = []
        #y_train = train_data[window_size:, :]
        y_train = []
        for i in range(window_size, len(train_data)):
            xset = []
            for j in range(train_data.shape[1]):
                a = train_data[i - window_size:i, j]
                xset.append(a)
            x_train.append(xset)
            y_train.append(train_data[i, :])

        # numpy arrayに変換
        x_train, y_train = np.array(x_train), np.array(y_train)
        # 訓練データのNumpy配列について、奥行を訓練データの数、行を60日分のデータ、列を抽出した株価データの種類数、の3次元に変換する
        # https://relaxing-living-life.com/147/
        x_train_3D = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], x_train.shape[2]))
        y_train_3D = np.reshape(y_train, (y_train.shape[0], y_train.shape[1]))

        self.model = Sequential()
        self.model.add(LSTM(units=50,return_sequences=True,input_shape=(x_train_3D.shape[1], window_size)))
        self.model.add(Dropout(0.2))
        self.model.add(LSTM(units=50,return_sequences=True))
        self.model.add(Dropout(0.2))
        self.model.add(LSTM(units=50,return_sequences=True))
        self.model.add(Dropout(0.2))
        self.model.add(LSTM(units=50))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(units=1))

        print('compile start (for the first time)')
        self.model.compile(optimizer='adam', loss='mean_squared_error')
        # TensorFlowのログを出力
        # 保存先ディレクトリがない場合は作成
        #import datetime
        #from pathlib import Path
        #dir_name = os.path.join(os.path.dirname(__file__), '../../../log/fit/', str(datetime.datetime.now().strftime("%Y%m%d-%H%M%S")))
        #dir = Path(dir_name)
        #dir.mkdir(parents=True, exist_ok=True)
        #tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=dir_name, histogram_freq=1)
        #history = self.model.fit(x_train, y_train, batch_size=32, epochs=100, verbose=0, callbacks=[tensorboard_callback])
        logger.info(y_train_3D.shape)
        history = self.model.fit(x_train_3D, y_train_3D, batch_size=32, epochs=100, verbose=0)
        
        # テストデータ(残り20%)作成
        test_data = self.scaled_Y[training_data_len - window_size:, :]
        #test_data = scaled_Y

        x_test = []
        #y_test = Y[training_data_len:, :]
        y_test = []
        for i in range(window_size, len(test_data)):
            xset = []
            for j in range(test_data.shape[1]):
                a = test_data[i - window_size:i, j]
                xset.append(a)
            x_test.append(xset)
            y_test.append(test_data[i, :])

        # numpy arrayに変換
        x_test = np.array(x_test)
        y_test = np.array(y_test)
        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], x_test.shape[2]))
        y_test = np.reshape(y_test, (y_test.shape[0], y_test.shape[1]))

        logger.info('x_test.shape:' + str(x_test.shape))
        #logger.info('y_test.shape:' + str(y_test.shape))
        
        predictions = self.model.predict(x_test)
        logger.info(str(self.code) + ' predictions.shape:' + str(predictions.shape))

        #predictions = self.scaler.inverse_transform(padding_array(predictions, Y.shape[1] - 1))
        predictions = self.scaler.inverse_transform(predictions)
        #logger.info('predictions.shape:' + str(predictions.shape))

        # MSR(平均二乗差)
        self.msr = np.mean((predictions - y_test) ** 2)
        return True

    def compile(self, delta_X: pd.Series, delta_Y: np.ndarray, last_date: pd.Timestamp):
        logger.info('compile [' + str(self.code) + ']' + str(delta_X))
        # 日数を結合
        self.days = pd.concat([self.days, delta_X])
        # どれくらいの期間をもとに予測するか
        window_size = 60
        # データ数が足りなければ終了
        # TODO:scaled_Yってどこで結合するんだっけ？一度データ数足りなかったら常にここでreturnされない？
        if len(self.scaled_Y) <= window_size:
            self.msr = 10000
            logger.info('[' + str(self.code) + ']cannot learn because few data')
            return
        training_data_begin = len(self.scaled_Y) - window_size
        Y = self.scaler.inverse_transform(self.scaled_Y)
        Y = np.concatenate([Y, delta_Y.to_numpy().reshape(-1, Y.shape[1])])
        # データを0-1に正規化
        self.scaled_Y = self.scaler.fit_transform(Y)

        if (self.model is None):
            print('model is none')
            self.first_compile(Y)
            return

        # 渡された差分の100%をトレーニングデータとして扱う
        training_data_len = int(np.ceil(len(Y) * 1.0))
        #training_data_len = int(np.ceil(len(Y) * 1.0))

        train_data = self.scaled_Y[training_data_begin:int(training_data_len), :]

        # train_dataをx_trainとy_trainに分ける
        x_train = []
        #y_train = train_data[window_size:, :]
        y_train = []
        for i in range(window_size, len(train_data)):
            xset = []
            for j in range(train_data.shape[1]):
                a = train_data[i - window_size:i, j]
                xset.append(a)
            x_train.append(xset)
#            y_train.append(train_data[i, 0])
            # TODO:これで6要素の予測いける？
            y_train.append(train_data[i, :])

        # numpy arrayに変換
        x_train, y_train = np.array(x_train), np.array(y_train)
        x_train_3D = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], x_train.shape[2]))
        # TODO:これで6要素の予測いける？
        y_train_3D = np.reshape(y_train, (y_train.shape[0], y_train.shape[1]))

#        history = self.model.fit(x_train_3D, y_train, batch_size=32, epochs=100, verbose=0)
        history = self.model.fit(x_train_3D, y_train_3D, batch_size=32, epochs=100, verbose=0)
        
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
        
    # TODO:正規化用にpaddingする
    def predict(self, days):
        window_size = 60
        x_test = []
        test_data = self.scaled_Y       # shape = (xxxx, 6) (6はclose, high, volume x 2)
        # 実データから作成(+1日まで)
        for i in range(window_size, len(test_data) + 1):
            xset = []
            for j in range(test_data.shape[1]):
                a = test_data[i - window_size:i, j]
                xset.append(a)
            x_test.append(xset)
        x_test = np.array(x_test)
        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], x_test.shape[2]))
        predictions = self.model.predict(x_test)
        #logger.info(str(predictions))
        logger.info('prediction shape[0]:' + str(predictions.shape[0]) + ' shape[1]:' + str(predictions.shape[1]))
        # モデルの予想値を含めて作成
        # TODO:特に理解怪しいからチェック
        for i in range(len(test_data) + 1, len(test_data) + days):
            test_data = np.append(test_data, predictions[-1])
            test_data = np.reshape(test_data, (test_data.shape[0], test_data.shape[1]))
            x_test_one = []
            x_test_one.append(test_data[i-window_size:i, :])
            x_test_one = np.array(x_test_one)
            x_test_one = np.reshape(x_test_one, (x_test_one.shape[0], x_test_one.shape[1], x_test_one.shape[2]))
            predictions = np.append(predictions, self.model.predict(x_test_one))
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