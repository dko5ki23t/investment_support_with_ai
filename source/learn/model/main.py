from . import model_1
from . import model_2
from . import model_3
from . import model_4
from . import model_5
from . import model_6
import pandas as pd

# 自作ロガー追加
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../logger'))
from logger import Logger
logger = Logger(__name__, 'model.log')


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


def estimate(data: pd.DataFrame, nikkei: pd.DataFrame, dir: str):
    code = data['code'].iloc[0]
    name = data['stock name'].iloc[0]
    days = data['day']
    closes = data['close']
    last_timestamp = data['timestamp'].iloc[-1]
    meta_data = {
        'version' : 1.0,
        'code' : code,
        'name' : name,
        'last_date' : last_timestamp,
        'model_num' : 4,
    }

    #data_nikkei = data.apply(extract_nikkei, args=(nikkei,), axis=1)
    # NaNを含む行は消す(TODO:これでいいのか)
    #data_nikkei = data_nikkei.dropna(how='any')


    model1 = model_1.model_1(
        code, name, days, closes, last_timestamp
    )
    model2 = model_2.model_2(
        code, name, days, closes, last_timestamp
    )
#    model3 = model_3.model_3(
#        code, name, days, data.filter(['close']).values, last_timestamp, dir
#    )
#    model4 = model_4.model_4(
#        code, name, days, data_nikkei.to_numpy(copy=True), last_timestamp,
#    )
#    model5 = model_5.model_5(
#        code, name, days, data[['close', 'high', 'volume']].to_numpy(copy=True), last_timestamp, dir
#    )
    model6 = model_6.model_6(
        code, name, days, closes, last_timestamp, dir
    )
    return [meta_data, model1, model2, model6]
    #return [meta_data, model5]
