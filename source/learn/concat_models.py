import argparse         # コマンドライン引数チェック用
import pandas as pd
import glob
from pathlib import Path
import numpy as np
import tqdm
import os
import model            # 学習モデル
import pickle


def set_argparse():
    parser = argparse.ArgumentParser(description='TODO')
    parser.add_argument('model1', help='機械学習の統計データがあるディレクトリ')
    parser.add_argument('model2', help='何回後の終値開示までに')
    parser.add_argument('out_dir', help='結果を保存するディレクトリ')
    args = parser.parse_args()
    return args

def main():
    args = set_argparse()
    # ファイル読み込み
    files = glob.glob(args.model1 + '/*.pkl')

    for index in tqdm.tqdm(range(len(files))):
        file = files[index]
        #basename = os.path.basename(file)
        #file2 = args.model1 + '/' + basename
        #out_file = args.out_dir + '/' + basename
        #models1 = pd.read_pickle(file)
        f = open(file, 'rb')
        models1 = pickle.load(f)
        f.close
        #f = open(file, 'rb')
        #models2 = pickle.load(f)
        #f.close
        #models1.pop(-1)
        if models1[4].name == 'model5':
            models1[4].tmp()
            print('OK')
        #f = open(out_file, 'wb')
        #pickle.dump(models1, f)
        #f.close
        f = open(file, 'wb')
        pickle.dump(models1, f)
        f.close

        #for e in models1:
        #    print(type(e))
        #for model in models1:
        #    print(type(model))
        #    print(model.code)
        #break
#        new_models = pd.concat([models1, models2])
#        new_models.to_pickle(args.out_dir + '/' + basename)
    

if __name__ == "__main__":
    main()