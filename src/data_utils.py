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

import random
import numpy as np
import torch
import pickle
import torch
import os
from torch.utils.data import (
    Dataset,
)
from datetime import datetime
import pandas as pd

import logging
logger = logging.getLogger(__name__)

## TEMP NAME
TRAIN_NAME = 'split_train_{}_{}.p'
VALID_NAME = 'split_valid_{}_{}.p'
AUGMENT_NAME = 'split_augment_{}_{}.p'

def save_pickle(path, data):
    with open(path, 'wb') as handle:
        pickle.dump(data, handle, protocol=4)

def load_pickle(path):
    with open(path, 'rb') as handle:
        return pickle.load(handle)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.device_count() > 0:
        torch.cuda.manual_seed_all(seed)

class IMDBDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_len=512, mlm_flag=False):

        assert os.path.isfile(file_path), 'there is no file please check again. [{}]'.format(file_path)

        self.max_len = max_len
        self.mlm_flag=mlm_flag

        dirname, filename = os.path.split(file_path)
        cache_filename = "cache_{}".format(filename)
        cache_path = os.path.join(dirname, cache_filename)
        if os.path.isfile(cache_path):
            logger.info("***** load cache dataset [{}] *****".format(cache_path))
            label, text, augment_text = load_pickle(cache_path)
        else:
            logger.info("***** tokenize dataset [{}] *****".format(file_path))

            data = load_pickle(file_path)
            label = data['label']
            text = data['clean_text']
            augment_text = data['backtranslated_text']

            logger.info("***** dataset size [{}] *****".format(str(len(text))))

            augment_text = [tokenizer.convert_tokens_to_ids(tokenizer.tokenize(t)) for t in augment_text]
            augment_text = [tokenizer.build_inputs_with_special_tokens(t) for t in augment_text]

            text = [tokenizer.convert_tokens_to_ids(tokenizer.tokenize(t)) for t in text]
            text = [tokenizer.build_inputs_with_special_tokens(t) for t in text]

            ## save tokenized file
            save_pickle(cache_path, (label, text, augment_text))

        self.augment_text = augment_text
        self.text = text
        self.label = label

    def __len__(self):
        return len(self.label)

    def __getitem__(self, item):
        ## real text
        if len(self.text[item]) > self.max_len:
            text = torch.tensor(
                [self.text[item][0]]
                + self.text[item][-(self.max_len - 1): -1]
                + [self.text[item][-1]],
                dtype=torch.long,
            )
        else:
            text = torch.tensor(self.text[item], dtype=torch.long)

        ## augmented text
        if len(self.augment_text[item]) > self.max_len:
            augment_text = torch.tensor(
                [self.augment_text[item][0]]
                + self.augment_text[item][-(self.max_len - 1): -1]
                + [self.augment_text[item][-1]],
                dtype=torch.long,
            )
        else:
            augment_text = torch.tensor(self.augment_text[item], dtype=torch.long)

        ## label
        label = torch.tensor(self.label[item], dtype=torch.long)

        if self.mlm_flag:
            return text

        return text, augment_text, label



class ResultWriter:
    def __init__(self, directory):

        self.dir = directory
        self.hparams = None
        self.load()
        self.writer = dict()

    def update(self, args, **results):
        now = datetime.now()
        date = "%s-%s %s:%s" % (now.month, now.day, now.hour, now.minute)
        self.writer.update({"date": date})
        self.writer.update(results)
        self.writer.update(vars(args))

        if self.hparams is None:
            self.hparams = pd.DataFrame(self.writer, index=[0])
        else:
            self.hparams = self.hparams.append(self.writer, ignore_index=True)
        self.save()

    def save(self):
        assert self.hparams is not None
        self.hparams.to_csv(self.dir, index=False)

    def load(self):
        path = os.path.split(self.dir)[0]
        if not os.path.exists(path):
            os.makedirs(path)
            self.hparams = None
        elif os.path.exists(self.dir):
            self.hparams = pd.read_csv(self.dir)
        else:
            self.hparams = None

