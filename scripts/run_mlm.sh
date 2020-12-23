#!/bin/bash

## Caution! Validation set must not included
## Only Unlabeled Dataset(for training) is used for MLM training.
## IMDB Labeled Dataset(for training) only contain 20 examples.
## So I think it is trivial to unuse Labeled Dataset

python src/train_mlm.py \
    --model_type=bert \
    --model_name_or_path=bert-base-uncased \
    --max_seq_length=128 \
    --do_train \
    --train_file=resource/uda_augment.p \
    --output_dir=pretrain_mlm \
    --save_steps=10000 \
    --fp16 \
    --per_device_train_batch_size=32 \
    --gradient_accumulation_steps=2 \
    --max_steps=100000 \
    --overwrite_output_dir