import pandas as pd
import numpy as np
import os
#os.environ['TF_CPP_MIN_LOG_LEVEL']='2'   # TensorFlowの警告を出力しない
os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private' # GPU占有化
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
import tensorflow as tf
import random
#tf.debugging.set_log_device_placement(True)

# 自作ロガー追加
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../../logger'))
from logger import Logger
logger = Logger(__name__, 'model_6.log')


# 乱数シードを固定
# TODO:再現性は出せなかった・・・ GPUの並列処理でどうしても誤差が出る？
# https://keras.io/getting_started/faq/#how-can-i-obtain-reproducible-results-using-keras-during-development
def set_seed(seed=1234):
    tf.keras.backend.clear_session()
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = '0'
    os.environ['TF_DETERMINISTIC_OPS'] = '1'

# どれくらいの期間をもとに予測するか
WINDOW_SIZE = 60

# ファイル名と同じクラスを持ち、共通のestimate関数を用意する
class model_6:
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
    def __init__(self, code: int, stock: str, X: pd.Series, Y: pd.Series, last_date: pd.Timestamp, dir: str, *discard):
        self.name = 'model6'
        self.code = code
        self.stock = stock
        self.days = X
        self.Y = Y
        self.last_date = last_date
        self.modelfile = dir + '/models/' + str(code) + '_' + self.name

        self.first_compile()

    # https://www.kaggle.com/code/kutaykutlu/time-series-tensorflow-rnn-lstm-introduction
    # LSTM用にデータセットを作る
    def windowed_dataset(self, series, window_size, batch_size, shuffle_buffer):
        series = tf.expand_dims(series, axis=-1)   # 最後に次元を一つ追加
        ds = tf.data.Dataset.from_tensor_slices(series)
        ds = ds.window(window_size + 1, shift=1, drop_remainder=True)  # TODO:window_size+1?
        ds = ds.flat_map(lambda w: w.batch(window_size + 1))
        #for element in ds:
        #    logger.info(element)
        ds = ds.shuffle(shuffle_buffer)
        ds = ds.map(lambda w: (w[:-1], w[1:]))
        ds = ds.batch(batch_size).prefetch(1)
        return ds
    
    # https://www.kaggle.com/code/kutaykutlu/time-series-tensorflow-rnn-lstm-introduction
    # LSTMモデルで予測
    # 戻り値：ndarray shape=(x, WINDOW_SIZE, 1)
    def model_forecast(self, model, series, window_size):
        series = tf.expand_dims(series, axis=-1)   # 最後に次元を一つ追加
        ds = tf.data.Dataset.from_tensor_slices(series)
        ds = ds.window(window_size, shift=1, drop_remainder=True)   # TODO:window_size?
        ds = ds.flat_map(lambda w: w.batch(window_size))
        ds = ds.batch(32).prefetch(1)
        forecast = model.predict(ds)
        return forecast
    
    # 学習に使用した値をダンプ
    # 同じデータを与えて同じ結果が得られるか確認するために使用
    

    def first_compile(self):
        # 乱数シード固定
        set_seed()
        # データ数が足りなければ終了
        # TODO:ここで終了するとmodelが作成されず、compile(),predict()呼び出し時にエラーになる
        if len(self.Y) <= WINDOW_SIZE + 1:
            self.msr = 10000
            logger.info('[' + str(self.code) + ']cannot learn because few data')
            return False

        # 訓練データとテストデータに分ける
        #x_train = self.Y[:training_data_len]
        x_train = self.Y

        shuffle_buffer_size = 1000  # TODO
        batch_size = 128            # TODO

        # データセット作成
        train_set = self.windowed_dataset(x_train, WINDOW_SIZE, batch_size, shuffle_buffer_size)
        logger.info('[' + str(self.code) + ']first compile')
        #for element in train_set.as_numpy_iterator():
        #    logger.info(element)
        #for element in train_set:
        #    logger.info(element)
        
        #model = Sequential([
        #    tf.keras.layers.Conv1D(filters=64, kernel_size=5,
        #                           strides=1, padding='causal',
        #                           activation='relu',
        #                           input_shape=[None, 1]),
        #    LSTM(64, return_sequences=True),
        #    LSTM(128, return_sequences=True),
        #    Dense(32, activation='relu'),
        #    Dense(16, activation='relu'),
        #    Dense(1),
        #    #tf.keras.layers.Lambda(lambda x: x * 400)
        #])

        model = Sequential([
            LSTM(units=50, return_sequences=True),
            Dropout(0.2),
            LSTM(units=50, return_sequences=True),
            Dropout(0.2),
            LSTM(units=50, return_sequences=True),
            Dropout(0.2),
            LSTM(units=50),
            Dropout(0.2),
            Dense(1),
        ])

        #model = Sequential()
        #model.add(LSTM(units=50,return_sequences=True,input_shape=(x_train.shape[1], 1)))
        #model.add(Dropout(0.2))
        #model.add(LSTM(units=50,return_sequences=True))
        #model.add(Dropout(0.2))
        #model.add(LSTM(units=50,return_sequences=True))
        #model.add(Dropout(0.2))
        #model.add(LSTM(units=50))
        #model.add(Dropout(0.2))
        #model.add(Dense(units=1))

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
        history = model.fit(train_set, epochs=100, verbose=0)
        #for w in model.weights:
        #    logger.info(w)

        # MSR(平均二乗差)
        #self.msr = np.mean((predictions - y_test) ** 2)
        self.msr = 0

        # モデル保存
        model.save(self.modelfile)
        return True

    def compile(self, delta_X: pd.Series, delta_Y: pd.Series, last_date: pd.Timestamp, *discard):
        logger.info('compile [' + str(self.code) + ']')
        # 日数を結合
        self.days = pd.concat([self.days, delta_X])
        # データを結合
        self.Y = pd.concat([self.Y, delta_Y])
        # データ数が足りなければ終了
        if len(self.Y) <= WINDOW_SIZE + 1:
            self.msr = 10000
            logger.info('[' + str(self.code) + ']cannot learn because few data')
            return
        
        # モデル読み込み
        model = tf.keras.models.load_model(self.modelfile)
        if (model is None):
            self.first_compile()
            return

        training_data_begin = len(self.Y) - (WINDOW_SIZE + 1)
        x_train = self.Y[training_data_begin:]
        shuffle_buffer_size = 1000  # TODO
        batch_size = 128            # TODO

        # データセット作成
        train_set = self.windowed_dataset(x_train, WINDOW_SIZE, batch_size, shuffle_buffer_size)
        logger.info('[' + str(self.code) + ']compile')
        #for element in train_set.as_numpy_iterator():
        #    logger.info(element)
        #for element in train_set:
        #    logger.info(element)

        history = model.fit(train_set, epochs=100, verbose=0)
        
        # モデル保存
        model.save(self.modelfile)

        # MSR(平均二乗差)
        # self.msr = np.mean((predictions - y_test) ** 2)

        self.last_date = last_date

        
    def predict(self, days):
        # モデル読み込み
        model = tf.keras.models.load_model(self.modelfile)
        test_data = self.Y
        predictions = self.model_forecast(model, test_data, WINDOW_SIZE)
        predictions = predictions[:, -1]  # shape=(x, 1)
        # モデルの予想値を含めて作成
        # TODO:days=1でしか試してないからこのfor文正しく動くかわからん
        for i in range(len(test_data) + 1, len(test_data) + days):
            test_data = test_data.append(pd.Series([predictions[-1]]))
            test_one = test_data[i-WINDOW_SIZE:i]
            predict_one = self.model_forecast(model, test_one, WINDOW_SIZE)
            predictions = np.append(predictions, predict_one[-1, -1])
            #predictions = np.reshape(predictions, (predictions.shape[0], 1))
        # 1次元化
        predictions = predictions.ravel()
        first_day = self.days.iloc[0] + WINDOW_SIZE
        last_day = self.days.iloc[-1] + WINDOW_SIZE + days
        ret_days = np.arange(first_day, last_day + 1, 1)
        return (ret_days, predictions)