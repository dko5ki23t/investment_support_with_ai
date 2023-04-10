from yahoo_finance_api2 import share
from yahoo_finance_api2.exceptions import YahooFinanceError
import pandas as pd
import sys
import plotly.express as px
from numpy.polynomial import Polynomial
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

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
df = result[1]
df['day from 5 years ago'] = range(0, len(df))
df['real/estimate'] = 'real'
df = df.drop(['timestamp', 'open', 'high', 'low', 'volume'], axis=1)
#print(df)

X = df['day from 5 years ago']
Y = df['close']

'''
回帰
閉じた式による導出
start
'''
# 重回帰関数を得る
#W = Polynomial.fit(X, Y, 1)
reg = make_pipeline(PolynomialFeatures(4), LinearRegression())
reg.fit(X.to_numpy().reshape(-1, 1), Y)
# 0-dayの最大値まで等間隔に配置した100個の数値を用意
x = np.linspace(0, len(df) - 1, 100)
# 目的変数の推定値を求める
#y_hat = W(x)
y_hat = reg.predict(x.reshape(-1, 1))
df_estimate = pd.DataFrame({'close':pd.Series(y_hat), 'day from 5 years ago':pd.Series(x), 'real/estimate':'estimate'})
df=pd.concat([df, df_estimate])

fig = px.line(df, x='day from 5 years ago', y='close', color='real/estimate')
fig.show()
# MSR(平均二乗差)
print('score:' + str(reg.score(X.to_numpy().reshape(-1, 1), Y)))
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
