#!/usr/bin/bash

# 各種パス読み取り
source ./path_info_bash.txt

python ../source/learn/learn.py $FETCHED_DIR $LEARNING_DIR
