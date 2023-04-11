import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

# ファイル名と同じクラスを持ち、共通のestimate関数を用意する
class model_1:

#    def __init__(self):
#        

    def name(self):
        return 'model1'

    """Fit the model.

        Fit all the transformers one after the other and transform the
        data. Finally, fit the transformed data using the final estimator.

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
        Y = data[str_y]

        '''
        回帰
        閉じた式による導出
        start
        '''
        # 4次の重回帰関数を得る
        #W = Polynomial.fit(X, Y, 1)
        reg = make_pipeline(PolynomialFeatures(4), LinearRegression())
        reg.fit(X.to_numpy().reshape(-1, 1), Y)
        # 0-dayの最大値まで等間隔に配置した100個の数値を用意
        #x = np.linspace(0, len(df) - 1, 100)
        # 目的変数の推定値を求める
        #y_hat = W(x)
        #y_hat = reg.predict(x.reshape(-1, 1))
        #df_estimate = pd.DataFrame({'close':pd.Series(y_hat), 'day from 5 years ago':pd.Series(x), 'real/estimate':'estimate'})
        #df=pd.concat([df, df_estimate])
        
        # MSR(平均二乗差)
        msr = reg.score(X.to_numpy().reshape(-1, 1), Y)

        # 次の日～n日後の値推定(TODO:全日はいらない？)
        #x = np.arange(X.iloc[-1] + 1, X.iloc[-1] + day_predict + 1, 1)
        #y_hat = reg.predict(x.reshape(-1, 1))
        
        return (reg, msr)
        '''
        回帰
        閉じた式による導出
        end
        '''

        '''
        回帰
        勾配降下法による導出
        start

        # 2列目の全部1は、次数0の分
        x_dash = np.vstack((X, np.ones_like(X))).T
        y = Y.to_numpy()
        w = np.zeros(x_dash.shape[1])

        #D = np.array([[1, 3], [3, 6], [6, 5], [8, 7]])
        #x_dash = np.vstack(([D[:,0], np.ones_like(D[:,0])])).T
        #Y = D[:,1]
        #w = np.zeros(x_dash.shape[1])

        max_epochs = 10000
        eta0 = 1e-10
        eps = 1e-4

        最急降下法
        for t in range(max_epochs):
            y_hat = x_dash @ w
            grad = 2 * x_dash.T @ (y_hat - y)
            #print(grad)
            if np.sum(np.abs(grad)) < eps:
                break
            w -= eta0 * grad
            #print(w)


        確率的勾配降下法
        for t in range(max_epochs):
            eta = eta0 / np.sqrt(1+t)
            i = np.random.randint(0, x_dash.shape[0])
            y_hat = np.dot(x_dash[i], w)
            grad = 2 * (y_hat - Y[i]) * x_dash[i]
            if np.sum(np.abs(grad)) < eps:
                break
            w -= eta * grad


        # 重み->多項式作成
        print(w)
        W = np.poly1d(w)
        print(W)
        x = np.linspace(0, len(df) - 1, 100)
        # 目的変数の推定値を求める
        y_hat_d = W(x)
        #y_hat_d = reg.predict(x.reshape(-1, 1))
        df_estimate = pd.DataFrame({'close':pd.Series(y_hat_d), 'day from 5 years ago':pd.Series(x), 'real/estimate':'estimate'})
        #df=pd.concat([df, df_estimate])

        fig = px.line(df_estimate, x='day from 5 years ago', y='close', color='real/estimate')
        #fig = px.line(df_estimate, x='day from 5 years ago', y='close', color='real/estimate')
        fig.show()

        回帰
        確率的勾配降下法による導出
        end
        '''