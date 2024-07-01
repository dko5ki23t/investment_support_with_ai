import argparse         # コマンドライン引数チェック用
import json
import sys
import polars as pl
import os
import datetime
from pathlib import Path
import glob
import math

# 自作ロガー追加
#import sys
#import os
#sys.path.append(os.path.join(os.path.dirname(__file__), '../logger'))
#from logger import Logger
#logger = Logger(__name__, 'analyze.log')

out_order_directory_default = os.path.join(os.path.dirname(__file__), '../../db/orders')

def set_argparse():
    parser = argparse.ArgumentParser(description='信頼度スコアが高い予想を元に売買指示を作成する')
    parser.add_argument('input', help='各予想ファイルが保存されたディレクトリ')
    parser.add_argument('-o', '--output', help='売買指示の出力ファイル', default='')
    args = parser.parse_args()
    return args

def main():
    args = set_argparse()
    # 出力先ファイル決定
    out_order_file = args.output
    if out_order_file == '':
        # 保存先ディレクトリがない場合は作成
        dir = Path(out_order_directory_default)
        dir.mkdir(parents=True, exist_ok=True)
        out_order_file = os.path.join(
            out_order_directory_default, 'order_1.json'
        )
    # 予想ファイル読み込み
    estimate_files = glob.glob(args.input + '/*.json')
    # 予想リスト
    estimates = []
    for estimate_file in estimate_files:
        with open(estimate_file) as f:
            estimates = estimates + json.load(f)["estimate"]
    # 売買指示リスト
    orders = []
    for estimate in estimates:
        if estimate['gains'] > 0:
            # 始値で買う
            order = {
                "date": estimate['date'],
                "due": estimate['date'],
                "code": estimate['code'],
                "type": "buy-open",
                "value": 0,
                "volume": 100
            }
            orders.append(order)
            # 買値との差がgainsを超えたら売る
            order = {
                "date": estimate['date'],
                "due": estimate['date'],
                "code": estimate['code'],
                "type": "sell-delta",
                "value": math.floor(estimate['gains']),
                "volume": 100
            }
            orders.append(order)
    output = {"orders": orders}
    # 売買指示をファイル出力
    with open(out_order_file, 'w') as f:
        json.dump(output, f, indent=2)


if __name__ == "__main__":
    main()
