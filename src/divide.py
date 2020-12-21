# coding=utf-8
# Copyright 2020 The JoungheeKim All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import torch
import os
import pickle
import numpy as np
import random
from sklearn.model_selection import train_test_split
from data_utils import save_pickle, load_pickle, set_seed

## trian name
TRAIN_NAME = 'split_train_{}_{}.p'
VALID_NAME = 'split_valid_{}_{}.p'
AUGMENT_NAME = 'split_augment_{}_{}.p'

def split_files(args):
    assert os.path.isfile(args.label_file), 'there is no label files, --label_file [{}]'.format(args.label_file)
    dirname, filename  = os.path.split(args.label_file)
    data = load_pickle(args.label_file)

    ## SPLIT data
    train_idx, leftover_idx, _, leftover_label = train_test_split(list(range(len(data['label']))), data['label'],train_size=args.labeled_data_size, stratify=data['label'])
    if len(leftover_idx) > args.valid_data_size:
        valid_idx, unlabel_idx, _, _ = train_test_split(leftover_idx, leftover_label, train_size=args.valid_data_size, stratify=leftover_label)
    else:
        valid_idx = leftover_idx
        unlabel_idx = []

    train_data = dict((key, np.array(item)[train_idx].tolist()) for key, item in zip(data.keys(), data.values()))
    valid_data = dict((key, np.array(item)[valid_idx].tolist()) for key, item in zip(data.keys(), data.values()))
    unlabel_data = dict((key, np.array(item)[unlabel_idx].tolist()) for key, item in zip(data.keys(), data.values()))

    if args.unlabel_file is not None and os.path.isfile(args.unlabel_file):
        additional_data = load_pickle(args.unlabel_file)
        for key in unlabel_data.keys():
            unlabel_data[key] += additional_data[key]

    if args.train_file is None:
        args.train_file = TRAIN_NAME.format(args.labeled_data_size, args.valid_data_size)
    train_path = os.path.join(args.output_dir, args.train_file)
    save_pickle(train_path, train_data)
    try:
        os.remove(os.path.join(args.output_dir, "cache_" + args.train_file))
    except:
        pass

    if args.valid_file is None:
        args.valid_file = VALID_NAME.format(args.labeled_data_size, args.valid_data_size)
    valid_path = os.path.join(args.output_dir, args.valid_file)
    save_pickle(valid_path, valid_data)
    try:
        os.remove(os.path.join(args.output_dir, "cache_" + args.valid_file))
    except:
        pass

    if args.augment_file is None:
        args.augment_file = AUGMENT_NAME.format(args.labeled_data_size, args.valid_data_size)
    augment_path = os.path.join(args.output_dir, args.augment_file)
    save_pickle(augment_path, unlabel_data)
    try:
        os.remove(os.path.join(args.output_dir, "cache_" + args.augment_file))
    except:
        pass

    args.train_file = train_path
    args.valid_file = valid_path
    args.augment_file = augment_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="Save Directory",
    )
    parser.add_argument(
        "--label_file",
        default=None,
        type=str,
        required=True,
        help="The input labeled data file (pickle).",
    )
    parser.add_argument(
        "--unlabel_file",
        default=None,
        type=str,
        help="The input unlabeled data file (pickle).",
    )
    parser.add_argument(
        "--labeled_data_size",
        default=20,
        type=int,
        help="labeled data size for train",
    )
    parser.add_argument(
        "--valid_data_size",
        default=3000,
        type=int,
        help="validation data size",
    )
    parser.add_argument(
        "--train_file",
        default=None,
        type=str,
        help="name of train file.",
    )
    parser.add_argument(
        "--valid_file",
        default=None,
        type=str,
        help="name of valid file.",
    )
    parser.add_argument(
        "--augment_file",
        default=None,
        type=str,
        help="name of augment file.",
    )
    parser.add_argument("--seed", type=int, default=9, help="random seed for initialization")
    args = parser.parse_args()

    ## Validation Check
    if args.labeled_data_size <= 0:
        raise ValueError(
            "labeled_data_size must exceed 0. Please check --labeled_data_size options [{}]".format(
                args.labeled_data_size
            )
        )
    if args.valid_data_size <= 0:
        raise ValueError(
            "labeled_data_size must exceed 0. Please check --valid_data_size options [{}]".format(
                args.valid_data_size
            )
        )

    if args.label_file is not None:
        if not os.path.isfile(args.label_file):
            raise ValueError(
                "There is no labeled file. Please check --label_file options [{}]".format(
                    args.label_file
                )
            )

    if args.unlabel_file is not None:
        if not os.path.isfile(args.unlabel_file):
            raise ValueError(
                "There is no unlabeled file. Please check --unlabel_file options [{}]".format(
                    args.unlabel_file
                )
            )

    ## make save path
    os.makedirs(args.output_dir, exist_ok=True)

    ## set Seed
    set_seed(args.seed)

    def _print_config(config):
        import pprint
        pp = pprint.PrettyPrinter(indent=4)
        pp.pprint(vars(config))

    _print_config(args)
    split_files(args)