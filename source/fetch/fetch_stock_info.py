import argparse
import sys
import json
import os

sys.path.append(os.path.dirname(__file__))
import core.fetch_stock_info_jquants as fetch_stock_info_jquants

settings_file = os.path.join(os.path.dirname(__file__), '../../settings/settings.json')
out_file_default = os.path.join(os.path.dirname(__file__), '../../db/stock_info.csv')

def set_argparse():
    parser = argparse.ArgumentParser(description='上場銘柄の情報をCSV形式で取得する')
    parser.add_argument('-o', '--out', help='保存先ファイル名', default='')
    args = parser.parse_args()
    return args

def main():
    args = set_argparse()
    try:
        settings = json.load(open(settings_file, 'r'))
    except:
        print('設定ファイルを読み込めませんでした')
        sys.exit(1)
    # 出力先ファイル決定
    out_file = args.out
    if out_file == '':
        out_file = out_file_default

    # 使用するサービスによって分岐
    using_service = settings['using_service']
    if using_service == 'jquants':
        fetch_stock_info_jquants.fetch_stock_info(out_file)
    else:
        print('設定ファイルに記載された株情報取得サービスでは上場銘柄の情報を取得できません')
        print('代わりにJQuantsを用いて情報を取得します')
        fetch_stock_info_jquants.fetch_stock_info(out_file)
        

if __name__ == "__main__":
    main()
    sys.exit()
