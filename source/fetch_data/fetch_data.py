from yahoo_finance_api2 import share
from yahoo_finance_api2.exceptions import YahooFinanceError
import pandas as pd
import sys

code = 3687 #ターゲット
S_year = 5 #取得年数
S_day = 1 #取得単位

# 目的変数を作成する
def kabuka():
    company_code = str(code) + '.T'
    my_share = share.Share(company_code)
    symbol_data = None

    try:
        symbol_data = my_share.get_historical(share.PERIOD_TYPE_YEAR,
                                              S_year,
                                              share.FREQUENCY_TYPE_DAY,
                                              S_day)
    except YahooFinanceError as e:
        print(e.message)
        sys.exit(1)
    # 株価をデータフレームに入れている
    df_base = pd.DataFrame(symbol_data)
    df_base = pd.DataFrame(symbol_data.values(), index=symbol_data.keys()).T
    df_base.timestamp = pd.to_datetime(df_base.timestamp, unit='ms')
    df_base.index = pd.DatetimeIndex(df_base.timestamp, name='timestamp').tz_localize('UTC').tz_convert('Asia/Tokyo')
    #df_base = df_base.drop(['timestamp', 'open', 'high', 'low', 'volume'], axis=1)
    
    #df_base = df_base.rename(columns={'close':company_code + '対象'})
    #df_base = df_base[:-1] #一番最後の行を削除
    df_base = df_base.reset_index(drop=True)
    
    
    return company_code, df_base

result = kabuka()
print(result[1])

# データフレームへもどす
df_base = result[1]

df_base.head()
