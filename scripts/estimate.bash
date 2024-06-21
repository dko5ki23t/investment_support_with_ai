#!/usr/bin/bash

# 各種パス読み取り
source ./path_info_bash.txt

if [ $# -lt 3 ]; then
    echo 'Usage : estimate.bat term now gain [use_fetched=Yes/No]'
    exit 1
fi

if [ "$4" == "Yes" ]; then
    python ../source/estimate/estimate.py -d $LEARNING_DIR -t $1 -n $2 -g $3 -a $ESTIMATE_DIR -u -f $FETCHED_DIR
else
    python ../source/estimate/estimate.py -d $LEARNING_DIR -t $1 -n $2 -g $3 -a $ESTIMATE_DIR -f $FETCHED_DIR
fi
