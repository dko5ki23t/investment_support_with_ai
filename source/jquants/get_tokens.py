import pandas as pd
import argparse
import sys
import json
import requests
import os

settings_file = os.path.join(os.path.dirname(__file__), '../../settings/jquants.json')

def set_argparse():
    parser = argparse.ArgumentParser(description='J-Quantsのトークン類を取得する')
    parser.add_argument('-m', '--mailaddress', help='J-Quantsに登録したメールアドレス。指定しない場合は'+settings_file+'のmailaddressを参照する', default='')
    parser.add_argument('-p', '--password', help='J-Quantsに登録したパスワード。指定しない場合は'+settings_file+'のpasswordを参照する', default='')
    args = parser.parse_args()
    return args

def get_id_token(refresh_token):
    r_post = requests.post(f"https://api.jquants.com/v1/token/auth_refresh?refreshtoken={refresh_token}")
    r_json = r_post.json()
    return r_json["idToken"]

def main():
    args = set_argparse()
    mailaddress = args.mailaddress  # メールアドレス
    password = args.password    # パスワード
    try:
        settings = json.load(open(settings_file, 'r'))
    except:
        print('設定ファイルを読み込めませんでした')
    # 引数で指定されていない場合は設定ファイルから読み込み
    if mailaddress == '':
        try:
            mailaddress = settings["mailaddress"]
        except:
            print('設定ファイルからメールアドレスを取得できませんでした')
            sys.exit(1)
    if password == '':
        try:
            password = settings["password"]
        except:
            print('設定ファイルからパスワードを取得できませんでした')
            sys.exit(1)
    
    # トークンID取得を試みる
    got_id_token = False
    print('トークンIDを取得しています・・・')
    try:
        id_token = get_id_token(settings["refresh_token"])
        settings["id_token"] = id_token
        print('トークンID取得に成功しました')
        got_id_token = True
    except:
        print('トークンID取得に失敗しました')
    
    if not got_id_token:
        # リフレッシュトークン取得を試みる
        print('リフレッシュトークンを取得しています・・・')
        data = {"mailaddress":mailaddress, "password":password}
        try:
            r_post = requests.post("https://api.jquants.com/v1/token/auth_user", data=json.dumps(data))
            r_json = r_post.json()
            settings["refresh_token"] = r_json["refreshToken"]
        except:
            print('リフレッシュトークン取得に失敗しました')
            sys.exit(1)
        
        # 再度トークンID取得を試みる
        print('トークンIDを取得しています・・・')
        try:
            id_token = get_id_token(settings["refresh_token"])
            settings["id_token"] = id_token
            print('トークンID取得に成功しました')
        except:
            print('トークンID取得に失敗しました')
            sys.exit(1)
    
    # 設定ファイル上書き
    with open(settings_file, 'w') as f:
        json.dump(settings, f, indent=2)
        print('取得したトークンを設定ファイルに書き込みました')

if __name__ == "__main__":
    main()
    sys.exit()
