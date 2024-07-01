import argparse
import sys
import json
import requests
import os
import polars as pl

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
import jquants.get_tokens

settings_file = os.path.join(os.path.dirname(__file__), '../../../settings/jquants.json')
out_file_default = os.path.join(os.path.dirname(__file__), '../../db/stock_info.csv')

def get_tokens(settings):
    jquants.get_tokens.get_tokens('', '', settings)
    settings = json.load(open(settings_file, 'r'))
    return settings["id_token"]

def fetch_stock_info_core(id_token: str):
    headers = {'Authorization': 'Bearer {}'.format(id_token)}
    r = requests.get("https://api.jquants.com/v1/listed/info", headers=headers)
    r_json = r.json()["info"]
    return pl.from_dicts(r_json)

def fetch_stock_info(out: str):
    try:
        settings = json.load(open(settings_file, 'r'))
    except:
        print('設定ファイルを読み込めませんでした')
        sys.exit(1)
    # 出力先ファイル決定
    out_file = out
    if out_file == '':
        out_file = out_file_default
    
    # IDトークン取得
    try:
        id_token = settings["id_token"]
    except:
        print('設定ファイルからIDトークンを取得できませんでした\nIDトークンを再取得します')
        id_token = get_tokens(settings)

    # 上場銘柄の情報取得を試みる
    print('銘柄情報を取得しています・・・')
    try:
        df = fetch_stock_info_core(id_token)
        print('銘柄情報取得に成功しました')
    except:
        # IDトークン取得からやり直し
        print('銘柄情報取得に失敗しました\nIDトークンを再取得します')
        id_token = get_tokens(settings)
        print('銘柄情報を取得しています・・・')
        df = fetch_stock_info_core(id_token)
    
    # 上場銘柄の情報を保存する
    df.write_csv(out_file, separator=',')
    print(f'取得した銘柄情報を{out_file}に書き込みました')

def set_argparse():
    parser = argparse.ArgumentParser(description='J-QuantsのAPIを用いて上場銘柄の情報をCSV形式で取得する')
    parser.add_argument('-o', '--out', help='保存先ファイル名', default='')
    args = parser.parse_args()
    return args

def main():
    args = set_argparse()
    fetch_stock_info(args.out)

if __name__ == "__main__":
    main()
    sys.exit()
