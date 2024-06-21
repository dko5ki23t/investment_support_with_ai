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

# 学習対象の目的変数の数
# 終値, 高値, 出来高, 日経平均株価：終値, 高値, 出来高
LEARNING_NUM = 6
# どれくらいの期間をもとに予測するか
WINDOW_SIZE = 60


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
        self.scaled_Y = self.scaler.fit_transform(Y.reshape(-1, LEARNING_NUM))

        self.first_compile(Y.reshape(-1, LEARNING_NUM))

    def first_compile(self, Y):
        # 終値と他の値を連結
        # TODO:Y_subのところ、きれいにしたい
        #self.scaled_Y = np.stack([Y_close, Y_sub[0], Y_sub[1], Y_sub[2], Y_sub[3], Y_sub[4]], axis=1)

        # 全体の80%をトレーニングデータとして扱う
        training_data_len = int(np.ceil(len(Y) * .8))
        # データ数が足りなければ終了
        # TODO:ここで終了するとmodelが作成されず、compile(),predict()呼び出し時にエラーになる
        if training_data_len <= WINDOW_SIZE:
            self.msr = 10000
            logger.info('[' + str(self.code) + ']cannot learn because few data')
            return False

        train_data = self.scaled_Y[0:int(training_data_len), :]

        # train_dataをx_trainとy_trainに分ける
        x_train, y_train = [], []
        for i in range(WINDOW_SIZE, len(train_data)):
            xset, yset = [], []
            for j in range(train_data.shape[1]):
                a = train_data[i - WINDOW_SIZE:i, j]
                xset.append(a)
                a = train_data[i, j]
                yset.append(a)
            x_train.append(xset)
            y_train.append(yset)

        # numpy arrayに変換
        x_train, y_train = np.array(x_train), np.array(y_train)
        x_train_3D = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], x_train.shape[2]))
        y_train = np.reshape(y_train, (y_train.shape[0], y_train.shape[1]))
        #logger.info(np.count_nonzero(np.isnan(x_train_3D)))
        #logger.info(x_train_3D.shape)

        self.model = Sequential()
        self.model.add(LSTM(units=50,return_sequences=True,input_shape=(x_train.shape[1], x_train_3D.shape[2])))
        self.model.add(Dropout(0.2))
        self.model.add(LSTM(units=50,return_sequences=True))
        self.model.add(Dropout(0.2))
        self.model.add(LSTM(units=50,return_sequences=True))
        self.model.add(Dropout(0.2))
        self.model.add(LSTM(units=50))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(units=x_train.shape[1]))

        print('[main]compile start (for the first time)')
        self.model.compile(optimizer='adam', loss='mean_squared_error')
        history = self.model.fit(x_train_3D, y_train, batch_size=32, epochs=100, verbose=0)
        
        # テストデータ(残り20%)作成
        test_data = self.scaled_Y[training_data_len - WINDOW_SIZE:, :]

        x_test, y_test = [], []
        for i in range(WINDOW_SIZE, len(test_data)):
            xset, yset = [], []
            for j in range(test_data.shape[1]):
                a = test_data[i - WINDOW_SIZE:i, j]
                xset.append(a)
                a = test_data[i, j]
                yset.append(a)
            x_test.append(xset)
            y_test.append(yset)

        # numpy arrayに変換
        x_test, y_test = np.array(x_test), np.array(y_test)
        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], x_test.shape[2]))
        y_test = np.reshape(y_test, (y_test.shape[0], y_test.shape[1]))
        
        predictions = self.model.predict(x_test)
        #predictions = self.scaler.inverse_transform(padding_array(predictions, LEARNING_SUB_NUM))
        predictions = self.scaler.inverse_transform(predictions)
        logger.info('[' + str(self.code) + ']predictions : ' + str(predictions))

        # MSR(平均二乗差)
        # TODO
        #self.msr = np.mean((predictions - y_test) ** 2)

        # -------サブ---------
        # サブは単一の説明変数としているので、model3と同様の処理を行う

        # 全体の80%をトレーニングデータとして扱う
        #training_data_lens = [0] * LEARNING_SUB_NUM
        #train_datas = [None] * LEARNING_SUB_NUM
        #x_trains = [None] * LEARNING_SUB_NUM
        #y_trains = [None] * LEARNING_SUB_NUM
#
        #for i in range(LEARNING_SUB_NUM):
        #    training_data_lens[i] = int(np.ceil(len(Y_sub[i]) * .8))
        #    # データ数が足りなければ終了
        #    if training_data_lens[i] <= WINDOW_SIZE:
        #        self.msr = 10000
        #        logger.info('cannot learn [' + str(self.code) + '] because few data')
        #        return False
#
        #    train_datas[i] = self.scaled_Y_sub[i][0:int(training_data_lens[i]), :]
#
        #    # train_datasをx_trainとy_trainに分ける
        #    x_trains[i] = []
        #    y_trains[i] = train_datas[i][WINDOW_SIZE:, :]
        #    for j in range(WINDOW_SIZE, len(train_datas[i])):
        #        x_trains[i].append(train_datas[i][j - WINDOW_SIZE:j, 0])
#
        #    # numpy arrayに変換
        #    x_trains[i], y_trains[i] = np.array(x_trains[i]), np.array(y_trains[i])
        #    x_trains[i] = np.reshape(x_trains[i], (x_trains[i].shape[0], x_trains[i].shape[1], 1))
#
        #    self.model_sub[i] = Sequential()
        #    self.model_sub[i].add(LSTM(units=50,return_sequences=True,input_shape=(x_trains[i].shape[1], 1)))
        #    self.model_sub[i].add(Dropout(0.2))
        #    self.model_sub[i].add(LSTM(units=50,return_sequences=True))
        #    self.model_sub[i].add(Dropout(0.2))
        #    self.model_sub[i].add(LSTM(units=50,return_sequences=True))
        #    self.model_sub[i].add(Dropout(0.2))
        #    self.model_sub[i].add(LSTM(units=50))
        #    self.model_sub[i].add(Dropout(0.2))
        #    self.model_sub[i].add(Dense(units=1))
#
        #    print('[sub(' + str(i) + ')]compile start (for the first time)')
        #    self.model_sub[i].compile(optimizer='adam', loss='mean_squared_error')
        #    history = self.model_sub[i].fit(x_trains[i], y_trains[i], batch_size=32, epochs=100, verbose=0)

            # サブのモデルの評価はしない
            # テストデータ(残り20%)作成
            #test_data = self.scaled_Ys[i][training_data_lens[i] - WINDOW_SIZE:, :]

            #x_test = []
            #y_test = []
            #for i in range(WINDOW_SIZE, len(test_data)):
            #    xset = []
            #    for j in range(test_data.shape[1]):
            #        a = test_data[i - WINDOW_SIZE:i, j]
            #        xset.append(a)
            #    x_test.append(xset)
            #    y_test.append(test_data[i, :])

            ## numpy arrayに変換
            #x_test, y_test = np.array(x_test), np.array(y_test)
            #x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], x_test.shape[2]))
            #y_test = np.reshape(y_test, (y_test.shape[0], y_test.shape[1]))
            #
            #prediction = self.model.predict(x_test)
            #logger.info(str(self.code) + ' prediction.shape:' + str(prediction.shape))
            #prediction = self.scaler.inverse_transform(self.padding_array(prediction, LEARNING_SUB_NUM - 1))
        
            # MSR(平均二乗差)
            #self.msr = np.mean((prediction - y_test) ** 2)

        return True

    # TODO
    def compile(self, delta_X: pd.Series, delta_Y: np.ndarray, last_date: pd.Timestamp):
        logger.info('compile [' + str(self.code) + ']' + str(delta_X))
        # 日数を結合
        self.days = pd.concat([self.days, delta_X])

        # データ数が足りなければ終了
        # TODO:scaled_Yってどこで結合するんだっけ？一度データ数足りなかったら常にここでreturnされない？
        if len(self.scaled_Y) <= WINDOW_SIZE:
            self.msr = 10000
            logger.info('[' + str(self.code) + ']cannot learn because few data')
            return
        training_data_begin = len(self.scaled_Y) - WINDOW_SIZE
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
        x_train, y_train = [], []
        for i in range(WINDOW_SIZE, len(train_data)):
            xset, yset = [], []
            for j in range(train_data.shape[1]):
                a = train_data[i - WINDOW_SIZE:i, j]
                xset.append(a)
                a = train_data[i, j]
                yset.append(a)
            x_train.append(xset)
            y_train.append(yset)

        # numpy arrayに変換
        x_train, y_train = np.array(x_train), np.array(y_train)
        x_train_3D = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], x_train.shape[2]))
        y_train = np.reshape(y_train, (y_train.shape[0], y_train.shape[1]))

#        history = self.model.fit(x_train_3D, y_train, batch_size=32, epochs=100, verbose=0)
        history = self.model.fit(x_train_3D, y_train, batch_size=32, epochs=100, verbose=0)

        # MSR(平均二乗差)
        # self.msr = np.mean((predictions - y_test) ** 2)
        self.last_date = last_date
        
    def predict(self, days):
        # 実データから作成(+1日まで)
        x_test = []
        test_data = self.scaled_Y       # shape = (xxxx, 6) (6はclose, high, volume x 2)
        logger.info(self.scaler.inverse_transform(test_data))
        # TODO:こんなにforループする必要ある？
        for i in range(WINDOW_SIZE, len(test_data) + 1):
            xset = []
            for j in range(test_data.shape[1]):
                a = test_data[i - WINDOW_SIZE:i, j]
                xset.append(a)
            x_test.append(xset)
        x_test = np.array(x_test)
        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], x_test.shape[2]))
        predictions = self.model.predict(x_test)
        #------サブ-------
        #x_test_sub = []
        #test_data_sub = self.scaled_Y_sub[0]
        #for i in range(WINDOW_SIZE, len(test_data_sub) + 1):
        #    x_test_sub.append(test_data_sub[i-WINDOW_SIZE:i, 0])
        #x_test_sub = np.array(x_test_sub)
        #x_test_sub = np.reshape(x_test_sub, (x_test_sub.shape[0], x_test_sub.shape[1], 1))
        #predictions_sub = self.model_sub[0].predict(x_test_sub)
        
        # モデルの予想値を含めて作成
        # TODO:特に理解怪しいからチェック。days=1ばっかりで試してるから実際は実行されてない
        for i in range(len(test_data) + 1, len(test_data) + days):
            test_data = np.append(test_data, predictions[-1])
            test_data = np.reshape(test_data, (test_data.shape[0], test_data.shape[1]))
            x_test_one = []
            for j in range(test_data.shape[1]):
                a = test_data[i - WINDOW_SIZE:i, j]
                x_test_one.append(a)
            #x_test_one.append(test_data[i-WINDOW_SIZE:i, :])
            x_test_one = np.array(x_test_one)
            x_test_one = np.reshape(x_test_one, (x_test_one.shape[0], x_test_one.shape[1], x_test_one.shape[2]))
            predictions = np.append(predictions, self.model.predict(x_test_one))
            predictions = np.reshape(predictions, (predictions.shape[0], predictions.shape[1]))
        # 正規化を元に戻す
        #y_hat = self.scaler.inverse_transform(padding_array(predictions, LEARNING_SUB_NUM))
        y_hats = self.scaler.inverse_transform(predictions)
        logger.info('predictions:' + str(y_hats))
        logger.info(y_hats[:, 0])
        # 1次元化
        y_hat = y_hats[:, 0].ravel()
        first_day = self.days.iloc[0] + WINDOW_SIZE
        last_day = self.days.iloc[-1] + WINDOW_SIZE + days
        ret_days = np.arange(first_day, last_day + 1, 1)
        return (ret_days, y_hat)