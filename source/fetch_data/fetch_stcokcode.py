import pandas as pd

url = 'https://www.jpx.co.jp/markets/statistics-equities/misc/tvdivq0000001vg2-att/data_j.xls'
df_jpx = pd.read_excel(url)
df_jpx_growth = df_jpx.loc[df_jpx["市場・商品区分"].str.contains('グロース')]
df_jpx_growth = df_jpx_growth.iloc[:, [1, 2, 3]]
df_jpx_growth.columns = ['code', 'name', 'division']
df_jpx_growth.to_csv("tosho_growth.csv")
