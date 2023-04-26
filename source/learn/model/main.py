from . import model_1
from . import model_2
from . import model_3
import pandas as pd

# 自作ロガー追加
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../logger'))
from logger import Logger
logger = Logger(__name__, 'model.log')

def estimate(data: pd.DataFrame, str_x, str_y):
    model1 = model_1.model_1(
        data['code'].iloc[0], data['stock name'].iloc[0],
        data[str_x], data[str_y], data['timestamp'].iloc[-1]
    )
    model2 = model_2.model_2(
        data['code'].iloc[0], data['stock name'].iloc[0],
        data[str_x], data[str_y], data['timestamp'].iloc[-1]
    )
    model3 = model_3.model_3(
        data['code'].iloc[0], data['stock name'].iloc[0],
        data[str_x], data.filter([str_y]).values, data['timestamp'].iloc[-1]
    )
    return (pd.Series([model1, model2, model3]))
