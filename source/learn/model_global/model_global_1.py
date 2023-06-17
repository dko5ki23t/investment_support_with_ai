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
logger = Logger(__name__, 'model_global_1.log')


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
class model_1:
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
    def __init__(self, days: pd.Series, df_closes_global: pd.DataFrame,
                 out_dir: str, last_date: pd.Timestamp, *discard):
        self.name = 'model_global_1'
        # データを加工
        logger.info(df_closes_global)
        dataset = df_closes_global.values
        logger.info(df_closes_global.values)
        self.data_mean = dataset.mean(axis=0)
        self.data_std = dataset.std(axis=0)
        dataset = (dataset-self.data_mean) / self.data_std
        self.columns = df_closes_global.columns
        self.dataset = dataset
        
        self.days = days
        self.last_date = last_date
        self.modelfile = out_dir + '/models/' + self.name

        self.first_compile()
    
    # https://www.kaggle.com/code/nicapotato/keras-timeseries-multi-step-multi-output
    def multivariate_multioutput_data(self, dataset, target, start_index, end_index,
                    history_size, target_size, step, single_step=False):
        data = []
        labels = []

        start_index = start_index + history_size
        if end_index is None:
            end_index = len(dataset) - target_size

        for i in range(start_index, end_index):
            indices = range(i-history_size, i, step)
            data.append(dataset[indices])

            if single_step:
                labels.append(target[i+target_size])
            else:
                labels.append(target[i:i+target_size])

        return np.array(data), np.array(labels)

    def first_compile(self):
        # 乱数シード固定
        set_seed()

        # データ数が足りなければ終了
        # TODO:ここで終了するとmodelが作成されず、compile(),predict()呼び出し時にエラーになる
        if len(self.dataset) <= WINDOW_SIZE:
            self.msr = 10000
            logger.info('cannot learn because few data')
            return False

        # 訓練データ作成
        logger.info(self.dataset)
        x_train, y_train = self.multivariate_multioutput_data(
                            self.dataset[:,:], self.dataset[:,:],
                            0, None, WINDOW_SIZE, 1, 1, single_step=True)

        logger.info(x_train.shape)

        shuffle_buffer_size = 1000  # TODO
        batch_size = 128            # TODO

        # tf.Dataデータセット作成
        train_set = tf.data.Dataset.from_tensor_slices((x_train, y_train))
        train_set = train_set.shuffle(shuffle_buffer_size)
        train_set = train_set.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
        logger.info('first compile')
        #for element in train_set.as_numpy_iterator():
        #    logger.info(element)
        #for element in train_set:
        #    logger.info(element)

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

    '''
    def compile(self, delta_closes: pd.DataFrame, last_date: pd.Timestamp, *discard):
        logger.info('compile (' + self.name + ')')
        # データを結合
        self.Y = pd.concat([self.Y, delta_Y])
        # 日数を更新
        self.days = pd.concat([self.days, delta_X])
        
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

        history = model.fit(train_set, epochs=100, verbose=0)
        
        # モデル保存
        model.save(self.modelfile)

        # MSR(平均二乗差)
        # self.msr = np.mean((predictions - y_test) ** 2)

        self.last_date = last_date
    '''
        
    def predict(self, days):
        # モデル読み込み
        model = tf.keras.models.load_model(self.modelfile)
        # 既存データの1日後を予測
        x_test, y_test = self.multivariate_multioutput_data(
                            self.dataset[:,:], self.dataset[:,:],
                            0, None, WINDOW_SIZE, 1, 1, single_step=True)
        predictions = model.predict(x_test)
        '''
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
        '''
        logger.info('predictions.shape:' + str(predictions.shape))
        # 正規化を元に戻す
        y_hat = predictions * self.data_std + self.data_mean
        logger.info('y_hat:' + str(y_hat.shape))
        
        # 1次元化
        #y_hat = y_hat.ravel()

        first_day = self.days.iloc[0] + WINDOW_SIZE
        last_day = self.days.iloc[-1] + WINDOW_SIZE + days
        ret_days = np.arange(first_day, last_day + 1, 1)
        return (ret_days, y_hat)