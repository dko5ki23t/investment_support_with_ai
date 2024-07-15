import argparse
import sys
import json
import os

sys.path.append(os.path.dirname(__file__))
import core.fetch_data_jquants as fetch_data_jquants
import core.fetch_data_yfinance as fetch_data_yfinance

settings_file = os.path.join(os.path.dirname(__file__), '../../settings/settings.json')
input_file_default = os.path.join(os.path.dirname(__file__), '../../db/stock_info.csv')
out_dir_default = os.path.join(os.path.dirname(__file__), '../../db/stock_data')

    
def set_argparse():
    parser = argparse.ArgumentParser(description='上場銘柄の株価データを取得する')
    parser.add_argument('-i', '--input', help='銘柄情報が記載されたCSVファイル', default='')
    parser.add_argument('-o', '--output', help='株価データ保存先ディレクトリ', default='')
    parser.add_argument('-c', '--codes', help='取得する銘柄の証券コード。複数指定可。', action='append')
    args = parser.parse_args()
    return args

def fetch_data(input: str, output: str, codes: list):
    try:
        settings = json.load(open(settings_file, 'r'))
    except:
        print('設定ファイルを読み込めませんでした')
        sys.exit(1)
    # 入力ファイル決定
    input_file = input
    if input_file == '':
        input_file = input_file_default
    # 出力先ディレクトリ決定
    out_dir = output
    if out_dir == '':
        out_dir = out_dir_default

    # 使用するサービスによって分岐
    using_service = settings['using_service']
    if using_service == 'jquants':
        fetch_data_jquants.fetch_data(input_file, out_dir, codes)
    elif using_service == 'yfinance':
        fetch_data_yfinance.fetch_data(input_file, out_dir, codes)
    else:
        print('設定ファイルに記載された株情報取得サービスでは上場銘柄の株価データを取得できません')
        print('代わりにJQuantsを用いて情報を取得します')
        fetch_data_jquants.fetch_data(input_file, out_dir)

# generator
def fetch_data_gen(input: str, output: str, codes: list):
    try:
        settings = json.load(open(settings_file, 'r'))
    except:
        print('設定ファイルを読み込めませんでした')
        sys.exit(1)
    # 入力ファイル決定
    input_file = input
    if input_file == '':
        input_file = input_file_default
    # 出力先ディレクトリ決定
    out_dir = output
    if out_dir == '':
        out_dir = out_dir_default

    # 使用するサービスによって分岐
    using_service = settings['using_service']
    if using_service == 'jquants':
        yield from fetch_data_jquants.fetch_data_gen(input_file, out_dir, codes)
    elif using_service == 'yfinance':
        yield from fetch_data_yfinance.fetch_data_gen(input_file, out_dir, codes)
    else:
        print('設定ファイルに記載された株情報取得サービスでは上場銘柄の株価データを取得できません')
        print('代わりにJQuantsを用いて情報を取得します')
        yield from fetch_data_jquants.fetch_data_gen(input_file, out_dir)

def main():
    args = set_argparse()
    fetch_data(args.input, args.output, args.codes)
    

if __name__ == "__main__":
    main()
    sys.exit()
