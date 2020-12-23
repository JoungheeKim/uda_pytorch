#!/bin/bash

LABEL_BATCH_SIZE=32
EVAL_BATCH_SIZE=64


TEST_FILE=resource/test.p
OUTPUT_DIR=output
EXPERIMENT_DIR=experiments/experiment.csv

LOGGING_STEPS=10
TRAIN_MAX_LEN=128
EVAL_MAX_LEN=128
MAX_STEP=3000

SEED=9
LEARNING_RATE=3e-5
TRAIN_FILE=resource/uda_train.p
VALID_FILE=resource/uda_valid.p
TEST_FILE=resource/test.p
PRETRAINED_MODEL=pretrain_mlm

python src/train.py \
    --output_dir=${OUTPUT_DIR} \
    --experiments_dir=${EXPERIMENT_DIR} \
    --do_train \
    --label_batch_size=${LABEL_BATCH_SIZE} \
    --eval_batch_size=${EVAL_BATCH_SIZE} \
    --overwrite_output_dir \
    --logging_steps=${LOGGING_STEPS} \
    --train_max_len=${TRAIN_MAX_LEN} \
    --eval_max_len=${EVAL_MAX_LEN} \
    --do_eval \
    --max_steps=${MAX_STEP} \
    --train_file=${TRAIN_FILE} \
    --valid_file=${VALID_FILE} \
    --test_file=${TEST_FILE} \
    --seed=${SEED} \
    --learning_rate=${LEARNING_RATE} \
    --pretrained_model=${PRETRAINED_MODEL} \
    --fp16



