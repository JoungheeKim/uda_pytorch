#!/bin/bash

## This bash commend include three process
## [1] download IMDB dataset
## [2] backtranslate IMDB dataset
## [3] save results

python src/back_translation.py \
    --save_dir=resource \
    --src2tgt_model=transformer.wmt19.en-de.single_model \
    --tgt2src_model=transformer.wmt19.de-en.single_model \
    --bpe=fastbpe \
    --tokenizer=moses \
    --batch_size=32 \
    --max_len=300 \
    --temperature=0.9 \
    --seed=1