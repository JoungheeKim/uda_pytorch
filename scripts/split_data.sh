#!/bin/bash

## This bash commend include three process
## [1] load labeled & unlabeled datasets
## [2] split data into three parts
##      A. train data
##      B. valid data
##      C. augment data
## [3] save results

python src/divide.py \
    --output_dir=resource \
    --label_file=resource/train.p \
    --unlabel_file=resource/unsupervised.p \
    --labeled_data_size=20 \
    --valid_data_size=3000 \
    --train_file=uda_train.p \
    --valid_file=uda_valid.p \
    --augment_file=uda_augment.p \
    --seed=9