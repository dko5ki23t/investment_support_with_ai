import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

# 自作ロガー追加
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../logger'))
from logger import Logger
logger = Logger(__name__, 'model_1.log')

# ファイル名と同じクラスを持ち、共通のestimate関数を用意する
class model_1:
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
    def __init__(self, code, stock, X, Y):
        self.name = 'model1'
        self.code = code
        self.stock = stock
        self.days = X
        '''
        回帰
        閉じた式による導出
        start
        '''
        # 4次の重回帰関数を得る
        #W = Polynomial.fit(X, Y, 1)
        self.reg = make_pipeline(PolynomialFeatures(4), LinearRegression())
        self.reg.fit(X.to_numpy().reshape(-1, 1), Y)
        # 0-dayの最大値まで等間隔に配置した100個の数値を用意
        #x = np.linspace(0, len(df) - 1, 100)
        # 目的変数の推定値を求める
        #y_hat = W(x)
        #y_hat = reg.predict(x.reshape(-1, 1))
        #df_estimate = pd.DataFrame({'close':pd.Series(y_hat), 'day from 5 years ago':pd.Series(x), 'real/estimate':'estimate'})
        #df=pd.concat([df, df_estimate])
        
        # MSR(平均二乗差)
        self.msr = self.reg.score(X.to_numpy().reshape(-1, 1), Y)

        # 次の日～n日後の値推定(TODO:全日はいらない？)
        #x = np.arange(X.iloc[-1] + 1, X.iloc[-1] + day_predict + 1, 1)
        #y_hat = reg.predict(x.reshape(-1, 1))

    def predict(self, days):
        first_day = self.days.iloc[0]
        last_day = self.days.iloc[-1] + days
        ret_days = np.arange(first_day, last_day + 1, 1)
        logger.info(ret_days)
        y_hat = self.reg.predict(ret_days.reshape(-1, 1))
        return (ret_days, y_hat)

#        np.append
