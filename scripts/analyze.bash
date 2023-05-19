#!/usr/bin/bash

# 各種パス読み取り
source ./path_info_bash.txt

python ../source/analyze/analyze.py -f $FETCHED_DIR -a $ESTIMATE_DIR
