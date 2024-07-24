import argparse         # コマンドライン引数チェック用
import json
import sys
import polars as pl
import os
import datetime
import jpholiday
import copy
import math
import tqdm
import bisect
import statistics
import plotly.express as px

# 自作ロガー追加
sys.path.append(os.path.join(os.path.dirname(__file__), '../logger'))
from logger import Logger
logger = Logger(__name__, 'evaluate.log')

stock_data_dir_default = os.path.join(os.path.dirname(__file__), '../../db/stock_data')

def d_to_str(d: datetime.datetime):
    return datetime.datetime.strftime(d, "%Y-%m-%d")

def str_to_d(s: str):
    return datetime.datetime.strptime(s, "%Y-%m-%d")

class HoldingStock:
    hold_from = ""  # 保持した（買った）日
    volume = 0      # 株数
    value_avg = 0   # 平均取得価格
    valuation = 0   # 評価損益

    def print(self):
        print(f'保持した日：{self.hold_from}\n株数：{self.volume}\n平均取得価格：{self.value_avg}\n評価損益：{self.valuation}')

# start~endの営業日リストを返す
def business_day_list(start: str, end: str):
    cur_date = str_to_d(start)
    end_date = str_to_d(end)
    ret = []
    while cur_date <= end_date:
        if cur_date.weekday() < 5 and not jpholiday.is_holiday(cur_date):
            ret.append(cur_date)
        cur_date = cur_date + datetime.timedelta(days=1)        
    return ret

# 日付文字列->独自の数字（ソート用）
def date_to_num(date: str):
    d_date = str_to_d(date)
    return ((d_date.year - 1900) * 10000 + d_date.month * 100 + d_date.day)

# 独自の数字->日付文字列（ソート用）
def num_to_date(num: int):
    y = math.floor(num / 10000) + 1900
    n = num % 10000
    m = math.floor(n / 100)
    d = n % 100
    return d_to_str(datetime.datetime(year=y, month=m, day=d))

# 日付->注文にアクセスできるデータベース
class OrdersDB:
    orders = []
    date_to_orders_idx = {}
    date_sorted_list = []    # date_to_orders_idxの日付を元に出した数値のソート済みリスト
    due_to_orders_idx = {}
    due_sorted_list = []    # due_to_orders_idxの日付を元に出した数値のソート済みリスト

    def __init__(self, orders: list):
        self.orders = orders
        # 注文リストのインデックスをdictに格納し、日付->注文にアクセスできるようにする
        for index in range(len(orders)):
            order = orders[index]
            if order['date'] in self.date_to_orders_idx:
                self.date_to_orders_idx[order['date']].append(index)
            else:
                self.date_to_orders_idx[order['date']] = [index]
                self.date_sorted_list.append(date_to_num(order['date']))
            if 'due2' in order:
                order_due = copy.copy(order['due2'])
            else:
                order_due = copy.copy(order['due'])
            if order_due == 'max':   # maxが締切日として指定されている場合は今日までとする
                order_due = d_to_str(datetime.datetime.today())
            if order_due in self.due_to_orders_idx:
                self.due_to_orders_idx[order_due].append(index)
            else:
                self.due_to_orders_idx[order_due] = [index]
                self.due_sorted_list.append(date_to_num(order_due))
        self.date_sorted_list.sort()
        self.due_sorted_list.sort()

    # 引数で指定した日付が開始日である注文のリストを返す
    def get_order_from_start_date(self, date: str):
        ret = []
        if date in self.date_to_orders_idx:
            indices = self.date_to_orders_idx[date]
            for index in indices:
                ret.append(self.orders[index])
        return ret
    
    # 引数で指定した日付で有効な（開始~締切の間の）注文のリストを返す
    def get_order_from_period(self, start: str, date: str):
        date_list = []
        due_list = []
        s_idx = bisect.bisect_left(self.date_sorted_list, date_to_num(start))
        e_idx = bisect.bisect_left(self.date_sorted_list, date_to_num(date))
        # start~dateの日付を注文日とする注文を返す
        for date_num in self.date_sorted_list[s_idx:e_idx+1]:
            date_list = date_list + self.date_to_orders_idx[num_to_date(date_num)]
        # 締切日dictのキーを日付若い順に並べたリストの内、引数の日付を挿入する位置を得る
        s_idx = bisect.bisect_left(self.due_sorted_list, date_to_num(date))
        # その位置以降の日付を締切日とする注文を返す
        for due_num in self.due_sorted_list[s_idx:]:
            due_list = due_list + self.due_to_orders_idx[num_to_date(due_num)]
        ret = []
        ret_set = (set(date_list) & set(due_list))
        for index in ret_set:
            ret.append(self.orders[index])
        return ret

'''
# start~end期間内での各period期間に対応する、
# 開始日・終了日の組とorderの配列のdictを返す
def get_orders_rolling(orders, start: str, end: str, period: int):
    ret = []
    all_days = business_day_list(start, end)
    start_ends = [[all_days[i], all_days[i+period-1]] for i in range(0, len(all_days)-period+1, 1)]
    # 日付->注文にアクセスできるデータベースを作成する
    orders_db = OrdersDB(orders)
    for start_end in start_ends:
        ret_orders = []
        orders_in_start_end = []   # 期間中の注文
        target_date = start_end[0]
        # 開始日～終了日までloop
        while target_date <= start_end[1]:
            orders_in_start_end = orders_in_start_end + orders_db.get_order_from_start_date(
                d_to_str(target_date))
            target_date = target_date + datetime.timedelta(days=1)
        # 期間中の注文でloop
        for order in orders_in_start_end:
            new_order = copy.deepcopy(order)
            # 締切日が範囲内に収まるようにする
            if order['due'] == 'max':
                new_order['due'] = d_to_str(start_end[1])
            else:
                new_order['due'] = d_to_str(min(str_to_d(order['due']), start_end[1]))
            ret_orders.append(new_order)
        ret.append({"start_and_end": start_end, "orders": ret_orders})
    return ret
'''

# start~end期間内での各period期間に対応する、
# 開始日・終了日の組を返す
def get_period_rolling(start: str, end: str, period: int):
    all_days = business_day_list(start, end)
    start_ends = [[all_days[i], all_days[i+period-1]] for i in range(0, len(all_days)-period+1, 1)]
    return start_ends

# 銘柄購入時の処理
def buy_stock(code: str, date: str, value: int, volume: int, holding_stocks: list, amount: int):
    '''
    volume 負数あり
    '''
    ret = amount
    # 購入できる最大株数
    volume_max = math.floor(amount / (value * 100)) * 100
    # 100株以上買える場合のみ続ける
    if volume_max >= 100:
        # 実際に購入する株数
        buy_volume = min(volume_max, volume)
        # 負数が指定されている場合は買えるだけ買う
        if volume < 0:
            buy_volume = volume_max
        # 保持している株に追加していく
        # 保持している株にある場合（買い増し）
        if code in holding_stocks:
            holding_stock = holding_stocks[code]
            holding_stock.hold_from = date
            holding_stock.value_avg = (
                (holding_stock.value_avg * holding_stock.volume) + (value * buy_volume)
            ) / (
                holding_stock.volume + buy_volume
            )
            holding_stock.volume = holding_stock.volume + buy_volume
            holding_stocks[code] = holding_stock
        # 保持している株にない場合（新規買い）
        else:
            holding_stock = HoldingStock()
            holding_stock.hold_from = date
            holding_stock.volume = buy_volume
            holding_stock.value_avg = value
            holding_stocks[code] = holding_stock
        # 買付可能額編集
        ret = amount - value * buy_volume
        # 買った記録をログ出力
        logger.info(f'{date}[{code}]買い {value} : {buy_volume}株 買付可能額 {ret}')
    return ret
    
# 銘柄売却時の処理
def sell_stock(code: str, date: str, value: int, volume: int, holding_stocks: list, amount: int):
    ret_amount = amount
    realized = 0
    # 保持している株から削除していく
    if code in holding_stocks:
        holding_stock = holding_stocks.pop(code)
        # 売却する株数
        sell_volume = min(holding_stock.volume, volume)
        # 負数が指定されている場合は売れるだけ売る
        if volume < 0:
            sell_volume = holding_stock.volume
        # 決定した売却株数から損益を追加
        realized = (value - holding_stock.value_avg) * sell_volume
        # 売却した株数を減算
        holding_stock.volume = holding_stock.volume - sell_volume
        # まだ株が残っているなら再度dictに追加
        if holding_stock.volume > 0:
            holding_stocks[code] = holding_stock
        # TODO: 上記時点でvalue_avgとか変化しないか？
        # 買付可能額編集
        ret_amount = amount + value * sell_volume
        # 売った記録をログ出力
        logger.info(f'{date}[{code}]売り {value} : {sell_volume}株 買付可能額 {ret_amount}')
    return {'amount': ret_amount, 'realized': realized}

# 指定された期間、元金でどのような損益結果が得られるかを評価する
def evaluate(orders_db: OrdersDB, stock_data_dir: str, base: int, gains: int, start: datetime.datetime, end: datetime.datetime):
    # 株価データdict(code:DataFrame)
    stock_data = {}
    # 所持している株のdict(code:HoldingStock)
    holding_stocks = {}
    # 実現損益
    realized_gains_loses = 0
    # 買付可能額
    amount = copy.copy(base)

    target_date = copy.copy(start)

    # 開始日～終了日までloop
    while target_date <= end:
        # 対象日に有効な注文のリスト取得
        target_orders = orders_db.get_order_from_period(d_to_str(start), d_to_str(target_date))
        # 各売買指示を処理
        for order in target_orders:
            # 対象銘柄を読み込み済みかどうか
            code = order['code']
            if code not in stock_data:
                # 株価データファイル読み込み
                data_file = os.path.join(stock_data_dir, f'{code}.parquet')
                try:
                    stock_data[code] = pl.read_parquet(data_file)
                except:
                    print('株価データの読み込みに失敗しました')
            # 対象銘柄のDataFrame
            stock_df = stock_data[code]
            # その対象日行
            target_term_df = stock_df.filter(
                pl.col('Date').str.strptime(pl.Date, "%Y-%m-%d") == target_date
            )
            order_val = order['value']
            # due2が設けられている場合かつdueを過ぎている場合(=due~due2の期間)は
            # value2の値を用いる
            if 'due2' in order and str_to_d(order['due']) < target_date:
                order_val = order['value2']
            # 買いの場合
            if order['type'] == 'buy':
                order_val = order['value']
                can_buy = target_term_df.filter(pl.col('Low') <= order_val)
                # 期間中に買値以下の値段になっているなら
                if len(can_buy) > 0:
                    # 高値が買値よりも低いなら高値で買う
                    buy_val = min(can_buy.get_column('High')[0], order_val)
                    amount = buy_stock(code, can_buy.get_column('Date')[0], buy_val, order['volume'], holding_stocks, amount)
            # 始値買いの場合
            elif order['type'] == 'buy-open':
                if len(target_term_df) > 0:
                    # 期間内最初の始値で買う
                    buy_val = target_term_df.get_column('Open')[0]
                    amount = buy_stock(code, target_term_df.get_column('Date')[0], buy_val, order['volume'], holding_stocks, amount)
            # 売りの場合
            elif order['type'] == 'sell':
                can_sell = target_term_df.filter(pl.col('High') >= order_val)
                # 期間中に売値以上の値段になっているなら
                if len(can_sell) > 0:
                    # 安値が売値よりも高いなら安値で売る
                    sell_val = max(can_sell.get_column('Low')[0], order_val)
                    ret = sell_stock(code, can_sell.get_column("Date")[0], sell_val, order['volume'], holding_stocks, amount)
                    amount = ret['amount']
                    realized_gains_loses = realized_gains_loses + ret['realized']
            # 差分売りの場合
            elif order['type'] == 'sell-delta':
                if code in holding_stocks:
                    goal = holding_stocks[code].value_avg + order_val
                    can_sell = target_term_df.filter(pl.col('High') >= goal)
                    # 期間中に保有株の平均取得値＋差分以上の値段になっているなら
                    if len(can_sell) > 0:
                        # 安値が売値よりも高いなら安値で売る
                        sell_val = max(can_sell.get_column('Low')[0], goal)
                        ret = sell_stock(code, can_sell.get_column("Date")[0], sell_val, order['volume'], holding_stocks, amount)
                        amount = ret['amount']
                        realized_gains_loses = realized_gains_loses + ret['realized']
        target_date = target_date + datetime.timedelta(days=1)
    # 評価損益計算
    for k, v in holding_stocks.items():
        # TODO: 以下、lenが0になる場合(大晦日等？)にどうするか問題
        if len(stock_data[k].filter(pl.col('Date') == d_to_str(end)).get_column('Close')) > 0:
            v.valuation = (stock_data[k].filter(pl.col('Date') == d_to_str(end)).get_column('Close')[0] - v.value_avg) * v.volume
    return {"holding_stocks": holding_stocks, "realized": realized_gains_loses}
    '''
    print("-----最終結果-----")
    print('[保有中の株]')
    for k, v in holding_stocks.items():
        print(f'コード：{k}')
        v.print()
        # TODO: 期間を指定してこのプログラムを実行する際は-1じゃない
        valuation = (stock_data[code].get_column('Close')[-1] - v.value_avg) * v.volume
        print(f'評価損益：{valuation}')
        print('-----')
    print('[実現損益]')
    print(realized_gains_loses)
    '''

def evaluate_gen(input: str, stock_data_dir='', start='2016-01-01', period=20, base=1000000, gains=100000):
    # 売買指示データ読み込み
    try:
        with open(input) as f:
            orders = json.load(f)
    except:
        print('売買指示ファイルの読み込みに失敗しました')
        sys.exit(1)
    # 株価データファイルが保存されたディレクトリ決定
    stock_data_dir_internal = stock_data_dir
    if stock_data_dir_internal == '':
        stock_data_dir_internal = stock_data_dir_default

    # start~本日までの、period期間分の開始日・終了日の組リストを得る
    start_ends = get_period_rolling(start, d_to_str(datetime.date.today()), period)
    # 各期間ごとに評価を行う
    total_period = len(start_ends)
    # 日付->注文にアクセスできるデータベースを作成する
    orders_db = OrdersDB(orders['orders'])
    success_num = 0     # 目標利益に達した数
    success_num2 = 0    # 目標利益に達した数（実現損益＋評価損益）
    realized_list = []  # 実現損益リスト
    realize_valuation_list = [] # 実現損益＋評価損益リスト
    for index in tqdm.tqdm(range(total_period)):
        valuation = 0
        start_and_end = start_ends[index]
        # 期間をログ出力
        logger.info(f'---{d_to_str(start_and_end[0])} ~ {d_to_str(start_and_end[1])}---')
        rets = evaluate(
            orders_db, stock_data_dir_internal, base, gains,
            start_and_end[0],
            start_and_end[1]
        )
        holding_stocks = rets["holding_stocks"]
        if holding_stocks is not None:
            for k, v in holding_stocks.items():
                logger.info(f"[{k}]評価損益：{v.valuation}")
                valuation = valuation + v.valuation
        realize_valuation_list.append(rets['realized'] + valuation)
        realized_list.append(rets['realized'])
        logger.info(f"実現損益：{rets['realized']}")
        if rets['realized'] >= gains:
            success_num = success_num + 1
        if rets['realized'] + valuation >= gains:
            success_num2 = success_num2 + 1
        yield [index, total_period+1]
    # 横軸=終了日時、縦軸=実現損益のグラフ作成
    ends = [e[1] for e in start_ends]
    fig_df = pl.DataFrame({'Date': ends, 'Realized': realized_list})
    fig_df = fig_df.with_columns(pl.lit('実現損益').alias('Name'))
    # 横軸=終了日時、縦軸=実現損益+評価損益のグラフ作成
    fig_df2 = pl.DataFrame({'Date': ends, 'Realized': realize_valuation_list})
    fig_df2 = fig_df2.with_columns(pl.lit('実現損益＋評価損益').alias('Name'))
    # 目標額の直線
    goal_df = pl.DataFrame({'Date': ends})
    goal_df = goal_df.with_columns(pl.lit(gains).cast(pl.Float64).alias('Realized'), pl.lit('目標損益').alias('Name'))
    # 連結
    fig_df = pl.concat([fig_df, fig_df2, goal_df])
    fig = px.line(x=fig_df['Date'], y=fig_df['Realized'], labels={'x': '期間終了日', 'y': '実現損益(円)'}, color=fig_df['Name'])
    fig.show()
    print(f'目標達成率：{(success_num / total_period) * 100}% ({success_num}/{total_period})')
    logger.info(f'目標達成率：{(success_num / total_period) * 100}% ({success_num}/{total_period})')
    print(f'平均：{statistics.mean(realized_list)} 最大：{max(realized_list)} 最小：{min(realized_list)} 中央：{statistics.median(realized_list)}')
    logger.info(f'平均：{statistics.mean(realized_list)} 最大：{max(realized_list)} 最小：{min(realized_list)} 中央：{statistics.median(realized_list)}')
    print(f'目標達成率（実現損益＋評価損益）：{(success_num2 / total_period) * 100}% ({success_num2}/{total_period})')
    logger.info(f'目標達成率（実現損益＋評価損益）：{(success_num2 / total_period) * 100}% ({success_num2}/{total_period})')
    print(f'平均：{statistics.mean(realize_valuation_list)} 最大：{max(realize_valuation_list)} 最小：{min(realize_valuation_list)} 中央：{statistics.median(realize_valuation_list)}')
    logger.info(f'平均：{statistics.mean(realize_valuation_list)} 最大：{max(realize_valuation_list)} 最小：{min(realize_valuation_list)} 中央：{statistics.median(realize_valuation_list)}')
    yield [total_period, total_period+1]

def set_argparse():
    parser = argparse.ArgumentParser(description='実データを用いて対象の売買を評価する')
    parser.add_argument('input', help='売買指示データが保存されたファイル')
    parser.add_argument('-d', '--data', help='株価データが保存されたディレクトリ', default='')
    parser.add_argument('-b', '--base', help='元金', type=int, default=1000000)
    parser.add_argument('-g', '--gains', help='目標利益', type=int, default=100000)
    parser.add_argument('-p', '--period', help='期間（日）', type=int, default=20)
    parser.add_argument('-s', '--start', help='評価を開始する日付（YYYY-MM-DD）', default='2016-01-01')
    args = parser.parse_args()
    return args

def main():
    args = set_argparse()
    for i in evaluate_gen(args.input, args.data, args.start, args.period, args.base, args.gains):
        pass
    sys.exit(0)

if __name__ == "__main__":
    main()
