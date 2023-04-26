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
    def __init__(self, code, stock, X, Y, last_date):
        self.name = 'model1'
        self.code = code
        self.stock = stock
        self.days = X
        self.last_date = last_date
        self.Y = Y
        '''
        回帰
        閉じた式による導出
        start
        '''
        # 4次の重回帰関数を得る
        self.reg = make_pipeline(PolynomialFeatures(4), LinearRegression())
        self.reg.fit(X.to_numpy().reshape(-1, 1), Y)
        
        # MSR(平均二乗差)
        self.msr = self.reg.score(X.to_numpy().reshape(-1, 1), Y)

    def compile(self, delta_X, delta_Y):
        # 日数,終値を結合
        self.days = pd.concat([self.days, delta_X])
        self.Y = pd.concat([self.Y, delta_Y])
        self.reg.fit(self.days.to_numpy().reshape(-1, 1), self.Y)
        # MSR(平均二乗差)
        self.msr = self.reg.score(self.days.to_numpy().reshape(-1, 1), self.Y)
        

    def predict(self, days):
        first_day = self.days.iloc[0]
        last_day = self.days.iloc[-1] + days
        ret_days = np.arange(first_day, last_day + 1, 1)
        logger.info(ret_days)
        y_hat = self.reg.predict(ret_days.reshape(-1, 1))
        return (ret_days, y_hat)

#        np.append
