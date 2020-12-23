#!/bin/bash

LABEL_BATCH_SIZE=16
UNLABEL_BATCH_SIZE=48
EVAL_BATCH_SIZE=64
OUTPUT_DIR=output
EXPERIMENT_DIR=experiments/experiment.csv

LOGGING_STEPS=20
TRAIN_MAX_LEN=128
EVAL_MAX_LEN=128
MAX_STEP=10000
ACCUMULATION_STEPS=1
SEED=9
LEARNING_RATE=2e-5
TRAIN_FILE=resource/uda_train.p
VALID_FILE=resource/uda_valid.p
AUGMENT_FILE=resource/uda_augment.p
TEST_FILE=resource/test.p
TSA=linear
CONFIDENCE_BETA=0.53
PRETRAINED_MODEL=pretrain_mlm

python src/train.py \
      --train_file=${TRAIN_FILE} \
      --valid_file=${VALID_FILE} \
      --augment_file=${AUGMENT_FILE} \
      --output_dir=${OUTPUT_DIR} \
      --experiments_dir=${EXPERIMENT_DIR} \
      --do_train \
      --label_batch_size=${LABEL_BATCH_SIZE} \
      --unlabel_batch_size=${UNLABEL_BATCH_SIZE} \
      --eval_batch_size=${EVAL_BATCH_SIZE} \
      --overwrite_output_dir \
      --logging_steps=${LOGGING_STEPS} \
      --train_max_len=${TRAIN_MAX_LEN} \
      --eval_max_len=${EVAL_MAX_LEN} \
      --do_eval \
      --max_steps=${MAX_STEP} \
      --test_file=${TEST_FILE} \
      --seed=${SEED} \
      --learning_rate=${LEARNING_RATE} \
      --do_uda \
      --tsa=${TSA} \
      --gradient_accumulation_steps=${ACCUMULATION_STEPS} \
      --fp16 \
      --pretrained_model=${PRETRAINED_MODEL} \
      --confidence_beta=${CONFIDENCE_BETA}