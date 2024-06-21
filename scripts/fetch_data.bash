#!/usr/bin/bash

# 各種パス読み取り
source ./path_info_bash.txt

python ../source/fetch_data/fetch_data.py $STOCKCODE_CSV $FETCHED_DIR 
