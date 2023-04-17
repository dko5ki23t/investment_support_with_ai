import pandas as pd
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
import plotly.express as px

# ファイル名と同じクラスを持ち、共通のestimate関数を用意する
class model_3:

#    def __init__(self):
#        

    def name(self):
        return 'model3'

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
    def estimate(self, data: pd.DataFrame, str_x, str_y):
        X = data[str_x]
        Y = data.filter([str_y]).values

        # データを0-1に正規化
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_Y = scaler.fit_transform(Y)
        # 全体の80%をトレーニングデータとして扱う
        #training_data_len = int(np.ceil(len(Y) * .8))
        training_data_len = int(np.ceil(len(Y) * 1.0))
        # どれくらいの期間をもとに予測するか
        window_size = 60

        train_data = scaled_Y[0:int(training_data_len), :]

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
        model.add(Dropout(0.2))
        model.add(LSTM(units=50,return_sequences=True))
        model.add(Dropout(0.2))
        model.add(LSTM(units=50,return_sequences=True))
        model.add(Dropout(0.2))
        model.add(LSTM(units=50))
        model.add(Dropout(0.2))
        model.add(Dense(units=1))

        print(str(data['stock name'].iloc[0]) + ':compile start')
        model.compile(optimizer='adam', loss='mean_squared_error')
        history = model.fit(x_train, y_train, batch_size=32, epochs=100)
        print(str(data['stock name'].iloc[0]) + ':compile end')
        
        # テストデータ(残り20%)作成
        #test_data = scaled_Y[training_data_len - window_size:, :]
        #test_data = scaled_Y

        #x_test = []
        #y_test = Y[training_data_len:, :]
        #for i in range(window_size, len(test_data)):
        #    x_test.append(test_data[i-window_size:i, 0])

        # numpy arrayに変換
        #x_test = np.array(x_test)
        #x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
        
        #predictions = model.predict(x_test)
        #predictions = scaler.inverse_transform(predictions)
        #df_estimate = pd.DataFrame({'close':pd.Series(predictions.ravel()), 'day from 5 years ago':pd.Series(X[training_data_len:]), 'real/estimate':'estimate'})
        #df_estimate = pd.DataFrame({'close':pd.Series(predictions.ravel()), 'day from 5 years ago':pd.Series(X), 'real/estimate':'estimate'})
        #df=pd.concat([data, df_estimate])

        #fig = px.line(df, x='day from 5 years ago', y='close', color='real/estimate')
        #fig.show()

        # MSR(平均二乗差)
        #mse = np.mean((predictions - Y[]) ** 2)

        # TODO:もうちょいうまくやる
        # return (model, mse, scaler)
        return (model, 0, [scaler, scaled_Y])