import argparse         # コマンドライン引数チェック用
import json
import os
from pathlib import Path
import glob
import math
from tqdm import tqdm
import polars as pl
import sys

# 自作ロガー追加
#import os
#sys.path.append(os.path.join(os.path.dirname(__file__), '../logger'))
#from logger import Logger
#logger = Logger(__name__, 'analyze.log')

stock_info_file_default = os.path.join(os.path.dirname(__file__), '../../db/stock_info.csv')
out_order_directory_default = os.path.join(os.path.dirname(__file__), '../../db/orders/order_1')

def name():
    """
    戦略の名前
    """
    return 'strategy1'

def version():
    """
    バージョン
    """
    return '1.0'

def description():
    """
    説明
    """
    return '各日利益が最大の銘柄1種を始値で買って予想利益分の差が出たら売る戦略で注文作成'

def strategy_gen(input: str, output='', base=1000000, stock_info_file='', filter_market_code=0, method_name=''):
    """
    【ジェネレータ】各日利益が最大の銘柄1種を始値で買って予想利益分の差が出たら売る戦略で注文作成

    input : str, default=''
            予想データが保存されたファイルまたはディレクトリ

    output : str, default=''
            注文の出力ファイル

    base : int, default=1000000
            元金。この値で買える範囲の銘柄のみを注文として出力する

    stock_info_file : str, default=''
            全銘柄の情報（銘柄名や市場コード等）が記載されたCSVファイル

    filter_market_code : int, default=0
            市場コードによるフィルタ。0の場合はフィルタリングしない

    method_name : str, default=''
            推定方法名によるフィルタ。空文字列の場合はフィルタリングしない

    戻り値 : list
            index0 : 処理が終了したインデックス。 index1 : 処理総数
    """
    # 出力先ファイル決定
    out_order_file = output
    if out_order_file == '':
        # 保存先ディレクトリがない場合は作成
        dir = Path(out_order_directory_default)
        dir.mkdir(parents=True, exist_ok=True)
        out_order_file = os.path.join(
            out_order_directory_default, f'order_{filter_market_code}_{method_name}.json'
        )
    # 銘柄情報ファイル読み込み
    stock_info_file_internal = stock_info_file
    if stock_info_file_internal == '':
        stock_info_file_internal = stock_info_file_default
    try:
        stock_info_df = pl.read_csv(stock_info_file_internal)
    except:
        print('銘柄情報データファイルの読み込みに失敗しました')
        sys.exit(1)
    # 予想ファイル読み込み
    estimate_files = glob.glob(input + '/*.json')
    # 予想リスト
    estimates = []
    print('予想ファイル読み込み・・・')
    for index in tqdm(range(len(estimate_files))):
        estimate_file = estimate_files[index]
        with open(estimate_file) as f:
            estimate_json = json.load(f)
        # 市場コードでフィルタリング
        code = estimate_json['code']
        market_code = stock_info_df.filter(pl.col('Code') == code).get_column('MarketCode')[0]
        if filter_market_code != 0 and market_code != filter_market_code:
            continue
        # 推定方法名でフィルタリング
        if method_name != '' and method_name != estimate_json['method_name']:
            continue
        estimates = estimates + estimate_json["estimate"]
    print('完了')
    # 売買指示リスト
    orders = []
    # 日付->予想のdict
    date_to_estimate = {}
    print('注文作成準備・・・')
    for index in tqdm(range(len(estimates))):
        estimate = estimates[index]
        gains = estimate['gains']
        date = estimate['date']
        if math.floor(estimate['gains']) > 0 and estimate['yest_close'] * 100 <= base:
            if date not in date_to_estimate or date_to_estimate[date]['gains'] < gains:
                # 同じ日の注文なら利益が大きい方のみ残す
                date_to_estimate[date] = estimate
    print('完了')
    print('注文作成・・・')
    # 注文作成
    # date_to_estimateを時系列でソート
    estimates = sorted(date_to_estimate.items(), key=lambda x: x[0])
    for index in tqdm(range(len(estimates))):
        estimate = estimates[index][1]
        # 始値で買う
        order = {
            "date": estimate['date'],
            "due": estimate['date'],
            "code": estimate['code'],
            "type": "buy-open",
            "value": 0,
            "volume": -1    # 買えるだけ買う
        }
        orders.append(order)
        # 買値との差がgainsを超えたら売る
        order = {
            "date": estimate['date'],
            "due": "max",
            "code": estimate['code'],
            "type": "sell-delta",
            "value": math.floor(estimate['gains']),
            "volume": -1    # 保持している分全て売る
        }
        orders.append(order)
        yield [index, len(estimates)+1]
    print('完了')
    print('ファイルへ出力・・・')
    output = {
        "name": name(),
        "version": version(),
        "base": base,
        "orders": orders
    }
    # 売買指示をファイル出力
    with open(out_order_file, 'w') as f:
        json.dump(output, f, indent=2)
    print('完了')
    yield [len(estimates), len(estimates)+1]

def set_argparse():
    parser = argparse.ArgumentParser(description='利益が高い予想を元に売買指示を作成する')
    parser.add_argument('input', help='各予想ファイルが保存されたディレクトリ')
    parser.add_argument('-b', '--base', help='元金。この値で買える銘柄のみを注文として出力する', type=int, default=1000000)
    parser.add_argument('-m', '--filter_market_code', help='inputで指定したディレクトリ内の予想データを市場コードでフィルタリング', type=int, default=0)
    parser.add_argument('--stock_info', help='全銘柄の情報（銘柄名や市場コード等）が記載されたCSVファイル', default='')
    parser.add_argument('--method_name', help='予想データのうち、推定方法名でフィルタリング', default='')
    parser.add_argument('-o', '--output', help='売買指示の出力ファイル', default='')
    args = parser.parse_args()
    return args

def main():
    args = set_argparse()
    for i in strategy_gen(args.input, args.output, args.base, args.stock_info, args.filter_market_code, args.method_name):
        pass

    sys.exit(0)


if __name__ == "__main__":
    main()
