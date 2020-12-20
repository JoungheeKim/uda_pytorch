import argparse
import torch
import os
import pickle
import logging
import random
import numpy as np
from sklearn.model_selection import train_test_split
from torch.nn.utils.rnn import pad_sequence
from typing import Dict, List, Tuple
from tqdm import tqdm, trange
from torch.nn import functional as F
import pandas as pd
from datetime import datetime
try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter

from torch.utils.data import (
    DataLoader,
    Dataset,
)
from transformers import (
    AdamW,
    BertForSequenceClassification,
    BertConfig,
    BertTokenizer,
    get_linear_schedule_with_warmup,
)
logger = logging.getLogger(__name__)

## trian name
TRAIN_NAME = 'split_train_{}_{}.p'
VALID_NAME = 'split_valid_{}_{}.p'
AUGMENT_NAME = 'split_augment_{}_{}.p'

def save_pickle(path, data):
    with open(path, 'wb') as handle:
        pickle.dump(data, handle, protocol=4)

def load_pickle(path):
    with open(path, 'rb') as handle:
        return pickle.load(handle)

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

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

    train_path = os.path.join(dirname, TRAIN_NAME.format(args.labeled_data_size, args.valid_data_size))
    save_pickle(train_path, train_data)
    try:
        os.remove(os.path.join(dirname, "cache_" + TRAIN_NAME.format(args.labeled_data_size, args.valid_data_size)))
    except:
        pass

    valid_path = os.path.join(dirname, VALID_NAME.format(args.labeled_data_size, args.valid_data_size))
    save_pickle(valid_path, valid_data)
    try:
        os.remove(os.path.join(dirname, "cache_" + VALID_NAME.format(args.labeled_data_size, args.valid_data_size)))
    except:
        pass

    augment_path = os.path.join(dirname, AUGMENT_NAME.format(args.labeled_data_size, args.valid_data_size))
    save_pickle(augment_path, unlabel_data)
    try:
        os.remove(os.path.join(dirname, "cache_" + AUGMENT_NAME.format(args.labeled_data_size, args.valid_data_size)))
    except:
        pass

    args.train_file = train_path
    args.valid_file = valid_path
    args.augment_file = augment_path

class IMDBDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_len):

        assert os.path.isfile(file_path), 'there is no file please check again. [{}]'.format(file_path)

        self.max_len = max_len

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

## add padding to make it parallel processing
def collate(data: List[torch.Tensor]):
    texts, augment_texts, labels = list(zip(*data))
    if tokenizer._pad_token is None:
        return (pad_sequence(texts, batch_first=True),
               pad_sequence(augment_texts, batch_first=True),
               torch.tensor(labels),
        )
    return (
        pad_sequence(texts, batch_first=True, padding_value=tokenizer.pad_token_id),
        pad_sequence(augment_texts, batch_first=True, padding_value=tokenizer.pad_token_id),
        torch.tensor(labels),
    )

def train(args, train_dataset, valid_dataset, unlabeled_dataset, model, tokenizer):
    tb_writer = SummaryWriter(comment=args.description)

    ## load dataloader
    train_dataloader = DataLoader(
        train_dataset, shuffle=True, batch_size=args.label_batch_size, collate_fn=collate
    )

    ## load loss function
    cross_entropy_fn = torch.nn.CrossEntropyLoss(reduction="none")
    def kl_divergence_fn(unlabeled_logits, augmented_logits, sharpen_ratio=1.0):
        ## https://github.com/huggingface/transformers/issues/1181

        loss_fn = torch.nn.KLDivLoss(reduction="none")
        return loss_fn(F.log_softmax(augmented_logits, dim=1), F.softmax(unlabeled_logits/sharpen_ratio, dim=1)).sum(dim=1)

    def get_tsa_threshold(global_step, t_total, num_labels, tsa='linear'):
        tsa = tsa.lower()
        if tsa == 'log':
            a_t = 1 - np.exp(-(global_step / t_total) * 5)
        elif tsa == 'exp':
            a_t = np.exp(-(1 / t_total) * 5)
        else:
            a_t = (global_step / t_total)

        threshold=a_t * (1-(1/num_labels)) + (1/num_labels)

        return threshold

    if args.max_steps > 0:
        t_total = args.max_steps

        args.num_train_epochs = (
            args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
        )
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    args.warmup_steps = int(args.warmup_percent * t_total)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use fp16 training."
            )
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
        model = torch.nn.DataParallel(model)

    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)
    logger.info("  Label Batch size = %d", args.label_batch_size)
    logger.info("  Label Data size = %d", len(train_dataset))

    step = 0
    global_step = 0
    best_loss = 1e10
    best_loss_step = 0
    best_acc = 0
    best_acc_step = 0
    stop_iter = False

    train_iterator = trange(0, int(args.num_train_epochs), desc="Epoch")

    ## if do eda
    if args.do_uda:
        unlabeled_dataloader = DataLoader(
            unlabeled_dataset, shuffle=True, batch_size=args.unlabel_batch_size, collate_fn=collate
        )
        unlabeled_iter = iter(unlabeled_dataloader)
        logger.info("  UnLabel Batch size = %d", args.unlabel_batch_size)
        logger.info("  UnLabel Data size = %d", len(unlabeled_dataset))


    model.zero_grad()
    for _ in train_iterator:
        for labeled_batch in train_dataloader:
            step += 1
            model.train()

            labeled_batch = tuple(t.to(args.device) for t in labeled_batch)
            labeled_texts, _, labels = labeled_batch

            label_outputs = model(input_ids=labeled_texts)
            ## get [CLS] token output
            label_outputs = label_outputs[0]
            cross_entropy_loss = cross_entropy_fn(label_outputs, labels)

            if args.tsa is not None:
                ## Get tsa Threshold
                tsa_threshold = get_tsa_threshold(global_step=global_step, t_total=t_total, num_labels=args.num_labels, tsa=args.tsa)

                ## selected Label prob
                label_prob = torch.exp(-cross_entropy_loss)

                ## selected pro less then threshold
                tsa_mask = label_prob.le(tsa_threshold)
                cross_entropy_loss = cross_entropy_loss * tsa_mask

            final_loss = cross_entropy_loss.mean()

            if args.do_uda:
                try:
                    unlabeled_batch = next(unlabeled_iter)
                except StopIteration:
                    # TODO check
                    unlabeled_iter = iter(unlabeled_dataloader)
                    unlabeled_batch = next(unlabeled_iter)
                unlabeled_batch = tuple(t.to(args.device) for t in unlabeled_batch)
                unlabeled_texts, augmented_texts, _ = unlabeled_batch

                augment_outputs = model(input_ids=augmented_texts)
                ## get [CLS] token output
                augment_outputs = augment_outputs[0]

                model.eval()
                with torch.no_grad():
                    unlabeled_outputs = model(input_ids=unlabeled_texts)
                    ## get [CLS] token output
                    unlabeled_outputs = unlabeled_outputs[0]
                model.train()

                consistency_loss = kl_divergence_fn(unlabeled_outputs, augment_outputs)

                ## Apply confidence beta
                unlabeled_prob = F.softmax(unlabeled_outputs).max(dim=1)[0]
                unlabeled_mask = unlabeled_prob.ge(args.confidence_beta)
                consistency_loss = consistency_loss * unlabeled_mask

                final_loss += args.uda_coeff * consistency_loss.mean()

            if args.gradient_accumulation_steps > 1:
                final_loss = final_loss / args.gradient_accumulation_steps
            if args.n_gpu > 1:
                final_loss = final_loss.mean()
            if args.fp16:
                with amp.scale_loss(final_loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                final_loss.backward()

            train_iterator.set_postfix_str(s="loss = {:.8f}".format(float(final_loss)), refresh=True)
            if step % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                model.zero_grad()
                global_step += 1

                if (args.logging_steps > 0 and global_step % args.logging_steps == 0 and global_step>0):
                #if (args.logging_steps > 0 and global_step % args.logging_steps == 0):
                    results = evaluate(args, valid_dataset, model, global_step)
                    eval_loss = results['loss']
                    eval_acc = results['accuracy']

                    if eval_loss<best_loss:
                        best_loss=eval_loss
                        best_loss_step = global_step

                        model_to_save = (
                            model.module if hasattr(model, "module") else model
                        )  # Take care of distributed/parallel training
                        model_to_save.save_pretrained(os.path.join(args.output_dir, 'loss'))
                        tokenizer.save_pretrained(os.path.join(args.output_dir, 'loss'))
                        torch.save(args, os.path.join(args.output_dir, 'loss', "training_args.bin"))


                    if eval_acc>best_acc:
                        best_acc=eval_acc
                        best_acc_step = global_step

                        model_to_save = (
                            model.module if hasattr(model, "module") else model
                        )  # Take care of distributed/parallel training
                        model_to_save.save_pretrained(os.path.join(args.output_dir, 'accuracy'))
                        tokenizer.save_pretrained(os.path.join(args.output_dir, 'accuracy'))
                        torch.save(args, os.path.join(args.output_dir, 'accuracy', "training_args.bin"))

                    logger.info("***** best_acc : %.4f *****", best_acc)
                    logger.info("***** best_loss : %.4f *****", best_loss)

                    for key, value in results.items():
                        tb_writer.add_scalar("eval_{}".format(key), value, global_step)


            if args.max_steps > 0 and global_step > args.max_steps:
                stop_iter = True
                break

        if stop_iter:
            break
    tb_writer.close()

    return {'best_valid_loss':best_loss,
            'best_valid_loss_step' : best_loss_step,
            'best_valid_acc':best_acc,
            'best_valid_acc_step': best_acc_step,
            }


def evaluate(args, test_dataset, model, global_step=0):
    test_dataloader = DataLoader(
        test_dataset, shuffle=False, batch_size=args.eval_batch_size, collate_fn=collate
    )

    if args.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
        model = torch.nn.DataParallel(model)

    logger.info("***** Running evaluation {}*****".format(global_step))
    eval_loss = 0.0
    nb_eval_steps = 0
    total_preds = []
    total_labels = []

    model.eval()
    for batch in tqdm(test_dataloader, desc="Evaluating"):
        batch = tuple(t.to(args.device) for t in batch)
        texts, _, labels = batch

        with torch.no_grad():
            outputs = model(input_ids=texts, labels=labels)
            loss, scores = outputs[:2]
            eval_loss += loss.mean().item()
        nb_eval_steps += 1

        preds = torch.softmax(scores, dim=1).detach().cpu().argmax(axis=1)
        labels = labels.detach().cpu()

        total_preds.append(preds)
        total_labels.append(labels)

    total_preds = torch.cat(total_preds)
    total_labels = torch.cat(total_labels)
    eval_loss = eval_loss/nb_eval_steps

    results = {
        "loss": eval_loss,
        "accuracy": (total_preds == total_labels).sum().item() / len(total_preds),
    }
    for key in sorted(results.keys()):
        logger.info("  %s = %s", key, str(results[key]))
    model.train()

    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--label_file",
        default=None,
        type=str,
        help="The input labeled data file (pickle).",
    )
    parser.add_argument(
        "--unlabel_file",
        default=None,
        type=str,
        help="The input unlabeled data file (pickle).",
    )
    parser.add_argument(
        "--train_file",
        default=None,
        type=str,
        help="The input training data file (pickle).",
    )
    parser.add_argument(
        "--valid_file",
        default=None,
        type=str,
        help="The input training data file (pickle).",
    )
    parser.add_argument(
        "--augment_file",
        default=None,
        type=str,
        help="The input training data file (pickle).",
    )
    parser.add_argument(
        "--test_file",
        default=None,
        type=str,
        help="The input training data file (pickle).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--experiments_dir",
        type=str,
        default='experiments/experiment.csv',
        help="The output directory where the model predictions and checkpoints will be written.",
    )

    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        help="The model checkpoint for weights initialization. Leave None if you want to train a model from scratch.",
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
        "--num_labels", default=2, type=int, help="Number of class labels.",
    )
    parser.add_argument(
        "--do_eval", action="store_true", help="Whether to run eval."
    )
    parser.add_argument(
        "--do_train", action="store_true", help="Whether to run train."
    )
    parser.add_argument(
        "--do_uda", action="store_true", help="Whether to run unsupervised Data Augmenation with Consistency training."
    )
    parser.add_argument(
        "--tsa", type=str, help="Whether to use tsa technique. Choose tsa type : linear, log, exp"
    )
    parser.add_argument(
        "--sharpen_ratio", type=float, default=1.0, help="Whether to use sharpening prediction for unlabled data, Choose value from 0 to 1"
    )
    parser.add_argument(
        "--confidence_beta", type=float, default=0.0,
        help="Whether to use confidence_beta for unlabled Data. Choose value from 0 to 1"
    )
    parser.add_argument(
        "--uda_coeff", type=float, default=1.0,
        help="loss ratio of supervised loss & unsupervised loss"
    )
    parser.add_argument(
        "--label_batch_size",
        default=32,
        type=int,
        help="Batch size for training.",
    )
    parser.add_argument(
        "--unlabel_batch_size",
        default=96,
        type=int,
        help="Batch size for training.",
    )
    parser.add_argument(
        "--eval_batch_size",
        default=64,
        type=int,
        help="Batch size for evaluation.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--train_max_len", default=512, type=int, help="Maximum sequence length.",
    )
    parser.add_argument(
        "--eval_max_len", default=512, type=int, help="Maximum sequence length.",
    )
    parser.add_argument(
        "--learning_rate", default=2e-5, type=float, help="The initial learning rate for Adam."
    )
    parser.add_argument(
        "--weight_decay", default=0.01, type=float, help="Weight decay if we apply some."
    )
    parser.add_argument(
        "--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer."
    )
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--num_train_epochs",
        default=3.0,
        type=float,
        help="Total number of training epochs to perform.",
    )
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    parser.add_argument(
        "--warmup_percent",
        default=0.1,
        type=float,
        help="Percentage of linear warmup over warmup_steps.",
    )
    parser.add_argument(
        "--logging_steps",
        default=20,
        type=int,
        help="Run evaluation during training at each logging step.",
    )
    parser.add_argument(
        "--pretrained_model",
        default=None,
        type=str,
        help="If there is pretrained Model",
    )
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument(
        "--overwrite_output_dir",
        action="store_true",
        help="Overwrite the content of the output directory",
    )
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
    parser.add_argument("--description", type=str, default='supervised learning', help="Tensorboard Description")
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )
    parser.add_argument(
        "--fp16_opt_level",
        type=str,
        default="O1",
        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
             "See details at https://nvidia.github.io/apex/amp.html",
    )
    parser.add_argument(
        "--local_rank", type=int, default=-1, help="For distributed training: local_rank"
    )
    args = parser.parse_args()

    ## GPU setting
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device

    # Set seed
    set_seed(args)

    if args.test_file is None and args.do_eval:
        raise ValueError(
            "Cannot do evaluation without an evaluation data file. Either supply a file to --test_file "
            "or remove the --do_eval argument."
        )
    if (
            os.path.exists(args.output_dir)
            and os.listdir(args.output_dir)
            and not args.overwrite_output_dir
    ):
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir
            )
        )
    if args.do_train:
        if args.label_file is None and (args.train_file is None or args.valid_file is None):
            raise ValueError(
                "Cannot do train without train and valid data. Either supply a file to --label_file or --train_file & --valid_file "
            )
        if args.do_uda and (args.augment_file is None and args.label_file is None):
            raise ValueError(
                "Cannot do train EDA training without augmented data. Either supply a file to --augment_file or --label_file to split data "
            )

        if args.train_file is None or args.valid_file is None:
            split_files(args)
        elif args.do_uda and args.augment_file is None:
            split_files(args)

    if args.do_eval:
        if args.do_train is None and args.pre_trained_model is None:
            raise ValueError(
                "Cannot do evalue without trained Model. Put a folder which contain trained Model to --pretrained_model"
            )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.warning(
        "Process device: %s, n_gpu: %s, 16-bits training: %s",
        device,
        args.n_gpu,
        args.fp16,
    )

    config = BertConfig.from_pretrained('bert-base-uncased' if args.model_name_or_path is None else args.model_name_or_path,
        num_labels=args.num_labels,
    )
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased' if args.model_name_or_path is None else args.model_name_or_path,)
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased' if args.model_name_or_path is None else args.model_name_or_path,
        config=config,
    )

    model.to(args.device)

    writer = ResultWriter(args.experiments_dir)
    results = {}

    if args.do_train:
        train_dataset = IMDBDataset(args.train_file, tokenizer, args.train_max_len)
        valid_dataset = IMDBDataset(args.valid_file, tokenizer, args.train_max_len)
        unlabeled_dataset = None
        if args.do_uda:
            unlabeled_dataset = IMDBDataset(args.augment_file, tokenizer, args.train_max_len)

        train_results = train(
            args, train_dataset, valid_dataset, unlabeled_dataset, model, tokenizer
        )
        results.update(**train_results)

        args.pretrained_model = args.output_dir

    if args.do_eval:
        test_dataset = IMDBDataset(args.test_file, tokenizer, args.eval_max_len)

        # Test LOSS
        loss_pretrained_model_path = os.path.join(args.pretrained_model, 'loss')
        model = BertForSequenceClassification.from_pretrained(loss_pretrained_model_path)
        model.to(args.device)
        test_results = evaluate(args, test_dataset, model)
        test_results.update(
            {
                'valid_check': 'loss',
            }
        )
        results.update(**test_results)
        writer.update(args, **results)

        # Test ACC
        acc_pretrained_model_path = os.path.join(args.pretrained_model, 'accuracy')
        model = BertForSequenceClassification.from_pretrained(acc_pretrained_model_path)
        model.to(args.device)
        test_results = evaluate(args, test_dataset, model)
        test_results.update(
            {
                'valid_check': 'accuracy',
            }
        )
        results.update(**test_results)
        writer.update(args, **results)










