from . import model_global_1
import pandas as pd
import glob
import datetime

# 自作ロガー追加
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../logger'))
from logger import Logger
logger = Logger(__name__, 'model_global.log')


# 同じ日付の日経平均株価情報を抽出して結合する
def extract_nikkei(row, nikkei: pd.DataFrame):
    target = nikkei[nikkei['date'] == row['date']]
    # 日経平均株価に該当データがなければ終了
    if len(target) == 0:
        return None
    target = target.iloc[-1]

    row = row[['close', 'high', 'volume']]  # closeを最初にすることが大事(model4のy_tarin作成時を参照)
    row['nikkei close'] = target['close']
    row['nikkei high'] = target['high']
    row['nikkei volume'] = target['volume']
    return row


def estimate_global(df_closes_global: pd.DataFrame, out_dir: str):
    days = pd.Series(range(len(df_closes_global)))
    last_timestamp = list(df_closes_global.index)[-1]
    logger.info(df_closes_global)

    meta_data = {
        'version' : 1.0,
        'last_date' : last_timestamp,
        'model_num' : 1,
    }

    model1 = model_global_1.model_1(
        days, df_closes_global, out_dir, last_timestamp
    )
    
    return [meta_data, model1]
