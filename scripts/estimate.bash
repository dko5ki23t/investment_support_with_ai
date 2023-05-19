#!/usr/bin/bash

# 各種パス読み取り
source ./path_info_bash.txt

if [ $# != 3 ]; then
    echo Usage : estimate.bat term now gain
    exit 1
fi

python ../source/estimate/estimate.py -d $LEARNING_DIR -t $1 -n $2 -g $3 -a $ESTIMATE_DIR
