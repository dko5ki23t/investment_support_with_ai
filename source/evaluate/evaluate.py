import argparse         # コマンドライン引数チェック用
import json
import sys
import polars as pl
import os
from datetime import datetime

# 自作ロガー追加
#import sys
#import os
#sys.path.append(os.path.join(os.path.dirname(__file__), '../logger'))
#from logger import Logger
#logger = Logger(__name__, 'analyze.log')

stock_data_dir_default = os.path.join(os.path.dirname(__file__), '../../db/stock_data')

class HoldingStock:
    hold_from = ""  # 保持した（買った）日
    volume = 0      # 株数
    value_avg = 0   # 平均取得価格

    def print(self):
        print(f'保持した日：{self.hold_from}\n株数：{self.volume}\n平均取得価格：{self.value_avg}')

def set_argparse():
    parser = argparse.ArgumentParser(description='実データを用いて対象の売買を評価する')
    parser.add_argument('input', help='売買指示データが保存されたファイル')
    parser.add_argument('-d', '--data', help='株価データが保存されたディレクトリ', default='')
    args = parser.parse_args()
    return args

def main():
    args = set_argparse()
    # 売買指示データ読み込み
    try:
        with open(args.input) as f:
            orders = json.load(f)
    except:
        print('売買指示ファイルの読み込みに失敗しました')
        sys.exit(1)
    # 株価データファイルが保存されたディレクトリ決定
    stock_data_dir = args.data
    if stock_data_dir == '':
        stock_data_dir = stock_data_dir_default

    # 株価データdict(code:DataFrame)
    stock_data = {}
    # 所持している株のdict(code:HoldingStock)
    holding_stocks = {}
    # 実現損益
    realized_gains_loses = 0
    # 各売買指示を処理
    for order in orders['orders']:
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
        order_date = datetime.strptime(order['date'], "%Y-%m-%d")
        order_due = datetime.strptime(order['due'], "%Y-%m-%d")
        # TODO:売りの場合は開始時が異なる
        # 売買ができる期間を抽出
        target_term_df = stock_df.filter(
            pl.col('Date').str.strptime(pl.Date, "%Y-%m-%d") >= order_date,
            pl.col('Date').str.strptime(pl.Date, "%Y-%m-%d") <= order_due,
        )
        # 買いの場合
        if order['type'] == 'buy':
            can_buy = target_term_df.filter(pl.col('Low') <= order['value'])
            # 期間中に買値以下の値段になっているなら
            if len(can_buy) > 0:
                # 高値が買値よりも低いなら高値で買う
                buy_val = min(can_buy.get_column('High')[0], order['value'])
                # 保持している株に追加していく
                # 保持している株にある場合（買い増し）
                if code in holding_stocks:
                    holding_stock = holding_stocks[code]
                    holding_stock.hold_from = can_buy.get_column('Date')[0]
                    holding_stock.value_avg = (
                        (holding_stock.value_avg * holding_stock.volume) + (buy_val * order['volume'])
                    ) / (
                        holding_stock.volume + order['volume']
                    )
                    holding_stock.volume = holding_stock.volume + order['volume']
                    holding_stocks[code] = holding_stock
                # 保持している株にない場合（新規買い）
                else:
                    holding_stock = HoldingStock()
                    holding_stock.hold_from = can_buy.get_column('Date')[0]
                    holding_stock.volume = order['volume']
                    holding_stock.value_avg = buy_val
                    holding_stocks[code] = holding_stock
                holding_stock.print()
        # 始値買いの場合
        elif order['type'] == 'buy-open':
            if len(target_term_df) > 0:
                # 期間内最初の始値で買う
                buy_val = target_term_df.get_column('Open')[0]
                # 保持している株に追加していく
                # 保持している株にある場合（買い増し）
                if code in holding_stocks:
                    holding_stock = holding_stocks[code]
                    holding_stock.hold_from = target_term_df.get_column('Date')[0]
                    holding_stock.value_avg = (
                        (holding_stock.value_avg * holding_stock.volume) + (buy_val * order['volume'])
                    ) / (
                        holding_stock.volume + order['volume']
                    )
                    holding_stock.volume = holding_stock.volume + order['volume']
                    holding_stocks[code] = holding_stock
                # 保持している株にない場合（新規買い）
                else:
                    holding_stock = HoldingStock()
                    holding_stock.hold_from = target_term_df.get_column('Date')[0]
                    holding_stock.volume = order['volume']
                    holding_stock.value_avg = buy_val
                    holding_stocks[code] = holding_stock
                holding_stock.print()
        # 売りの場合
        elif order['type'] == 'sell':
            can_sell = target_term_df.filter(pl.col('High') >= order['value'])
            # 期間中に売値以上の値段になっているなら
            if len(can_sell) > 0:
                # 安値が売値よりも高いなら安値で売る
                sell_val = max(can_sell.get_column('Low')[0], order['value'])
                # 保持している株から削除していく
                if code in holding_stocks:
                    holding_stock = holding_stocks.pop(code)
                    # 持っている株数分は全て売れるとし、損益を追加
                    realized_gains_loses = realized_gains_loses + (sell_val - holding_stock.value_avg) * holding_stock.volume
                    print(f'[{code}] 売り\n実現損益：{realized_gains_loses}')
        # 差分売りの場合
        elif order['type'] == 'sell-delta':
            if code in holding_stocks:
                goal = holding_stocks[code].value_avg + order['value']
                can_sell = target_term_df.filter(pl.col('High') >= goal)
                # 期間中に保有株の平均取得値＋差分以上の値段になっているなら
                if len(can_sell) > 0:
                    # 安値が売値よりも高いなら安値で売る
                    sell_val = max(can_sell.get_column('Low')[0], goal)
                    # 保持している株から削除していく
                    holding_stock = holding_stocks.pop(code)
                    # 持っている株数分は全て売れるとし、損益を追加
                    realized_gains_loses = realized_gains_loses + (sell_val - holding_stock.value_avg) * holding_stock.volume
                    print(f'[{code}] 売り\n実現損益：{realized_gains_loses}')
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

if __name__ == "__main__":
    main()
