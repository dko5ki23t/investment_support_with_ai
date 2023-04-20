import pandas as pd
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout

# 自作ロガー追加
import sys
import os
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
    def __init__(self, code, stock, X, Y):
        self.name = 'model3'
        self.code = code
        self.stock = stock
        self.days = X

        # データを0-1に正規化
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.scaled_Y = self.scaler.fit_transform(Y)
        # 全体の80%をトレーニングデータとして扱う
        training_data_len = int(np.ceil(len(Y) * .8))
        #training_data_len = int(np.ceil(len(Y) * 1.0))
        # どれくらいの期間をもとに予測するか
        window_size = 60

        train_data = self.scaled_Y[0:int(training_data_len), :]

        # train_dataをx_trainとy_trainに分ける
        x_train = []
        y_train = train_data[window_size:, :]
        for i in range(window_size, len(train_data)):
            x_train.append(train_data[i - window_size:i, 0])

        # numpy arrayに変換
        x_train, y_train = np.array(x_train), np.array(y_train)
        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

        logger.info('x_train.shape:' + str(x_train.shape))
        logger.info('y_train.shape:' + str(y_train.shape))

        self.model = Sequential()
        self.model.add(LSTM(units=50,return_sequences=True,input_shape=(x_train.shape[1], 1)))
        self.model.add(Dropout(0.2))
        self.model.add(LSTM(units=50,return_sequences=True))
        self.model.add(Dropout(0.2))
        self.model.add(LSTM(units=50,return_sequences=True))
        self.model.add(Dropout(0.2))
        self.model.add(LSTM(units=50))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(units=1))

        self.model.compile(optimizer='adam', loss='mean_squared_error')
        history = self.model.fit(x_train, y_train, batch_size=32, epochs=100)
        
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
        #df_estimate = pd.DataFrame({'close':pd.Series(predictions.ravel()), 'day from 5 years ago':pd.Series(X[training_data_len:]), 'real/estimate':'estimate'})
        #df_estimate = pd.DataFrame({'close':pd.Series(predictions.ravel()), 'day from 5 years ago':pd.Series(X), 'real/estimate':'estimate'})
        #df=pd.concat([data, df_estimate])

        #fig = px.line(df, x='day from 5 years ago', y='close', color='real/estimate')
        #fig.show()

        # MSR(平均二乗差)
        self.msr = np.mean((predictions - y_test) ** 2)

        # TODO:もうちょいうまくやる
        # return (model, mse, scaler)

    def predict(self, days):
        window_size = 60
        x_test = []
        test_data = self.scaled_Y
        # 実データから作成(+1日まで)
        for i in range(window_size, len(test_data) + 1):
            x_test.append(test_data[i-window_size:i, 0])
        x_test = np.array(x_test)
        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
        predictions = self.model.predict(x_test)
        # モデルの予想値を含めて作成
        for i in range(len(test_data) + 1, len(test_data) + days):
            test_data = np.append(test_data, predictions[-1])
            test_data = np.reshape(test_data, (test_data.shape[0], 1))
            x_test_one = []
            x_test_one.append(test_data[i-window_size:i, 0])
            x_test_one = np.array(x_test_one)
            x_test_one = np.reshape(x_test_one, (x_test_one.shape[0], x_test_one.shape[1], 1))
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