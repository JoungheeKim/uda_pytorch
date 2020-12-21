import argparse
import torch
import os
import json
import pickle
from nltk.tokenize import sent_tokenize
import nltk
nltk.download('punkt')
import datasets
from tqdm import tqdm
import numpy as np
import random

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.device_count() > 0:
        torch.cuda.manual_seed_all(seed)

def clean_web_text(st):
    """
    Adapted from UDA official code
    https://github.com/google-research/uda/blob/master/text/utils/imdb_format.py
    """
    st = st.replace("<br />", " ")
    st = st.replace("&quot;", '"')
    st = st.replace("<p>", " ")
    if "<a href=" in st:
        while "<a href=" in st:
            start_pos = st.find("<a href=")
            end_pos = st.find(">", start_pos)
            if end_pos != -1:
                st = st[:start_pos] + st[end_pos + 1 :]
            else:
                st = st[:start_pos] + st[start_pos + len("<a href=")]

        st = st.replace("</a>", "")
    st = st.replace("\\n", " ")
    return st


def split_sent_by_punc(sent, punc_list, max_len):
    """
    Adapted from UDA official code
    https://github.com/google-research/uda/blob/master/back_translate/split_paragraphs.py
    """

    if len(punc_list) == 0 or len(sent) <= max_len:
        return [sent]

    punc = punc_list[0]
    if punc == " " or not punc:
        offset = 100
    else:
        offset = 5

    sent_list = []
    start = 0
    while start < len(sent):
        if punc:
            pos = sent.find(punc, start + offset)
        else:
            pos = start + offset
        if pos != -1:
            sent_list += [sent[start: pos + 1]]
            start = pos + 1
        else:
            sent_list += [sent[start:]]
            break

    new_sent_list = []
    for temp_sent in sent_list:
        new_sent_list += split_sent_by_punc(temp_sent, punc_list[1:], max_len)

    return new_sent_list

def split_sent(content, max_len):
    """
    Adapted from UDA Official code
    https://github.com/google-research/uda/blob/master/back_translate/split_paragraphs.py
    """
    sent_list = sent_tokenize(content)
    new_sent_list  = []
    split_punc_list = [".", ";", ",", " ", ""]
    for sent in sent_list:
        new_sent_list += split_sent_by_punc(sent, split_punc_list, max_len)
    return new_sent_list, len(new_sent_list)

## batch iteration
def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]

## save pickle
def save_pickle(path, data):
    with open(path, 'wb') as handle:
        pickle.dump(data, handle)


def run(args):

    ## load translator
    src2tgt = torch.hub.load("pytorch/fairseq", args.src2tgt_model, tokenizer=args.tokenizer, bpe=args.bpe).to(
        args.device).eval()
    tgt2src = torch.hub.load("pytorch/fairseq", args.tgt2src_model, tokenizer=args.tokenizer, bpe=args.bpe).to(
        args.device).eval()

    ## load Dataset
    imdb_data = datasets.load_dataset('imdb')

    data_list = ['train', 'test', 'unsupervised']
    for dataname in tqdm(data_list, desc='data name'):
        temp_dataset = imdb_data[dataname]
        temp_docs = temp_dataset['text']
        temp_label = temp_dataset['label']

        ## clean web tag from text
        temp_docs = [clean_web_text(temp_sent) for temp_sent in temp_docs]
        new_contents = []
        new_contents_length = []
        for temp_doc in temp_docs:
            new_sents, new_sents_length = split_sent(temp_doc, args.max_len)
            new_contents += new_sents
            new_contents_length += [new_sents_length]

        backtranslated_contents = []
        for contents in tqdm(batch(new_contents, args.batch_size), total=int(len(new_contents)/args.batch_size)):
            with torch.no_grad():
                translated_data = src2tgt.translate(
                    contents,
                    sampling=True if args.temperature is not None else False,
                    temperature=args.temperature,
                )
                back_translated_data = tgt2src.translate(
                    translated_data,
                    sampling=True if args.temperature is not None else False,
                    temperature=args.temperature,
                )

            backtranslated_contents += back_translated_data

        merge_backtranslated_contents=[]
        merge_new_contents = []
        cumulate_length = 0
        for temp_length in new_contents_length:
            merge_backtranslated_contents += [" ".join(backtranslated_contents[cumulate_length:cumulate_length + temp_length])]
            merge_new_contents += [" ".join(new_contents[cumulate_length:cumulate_length + temp_length])]
            cumulate_length += temp_length

        save_data = {
            'raw_text' : temp_docs,
            'label' : temp_label,
            'clean_text' : merge_new_contents,
            'backtranslated_text' : merge_backtranslated_contents,
        }

        save_path = os.path.join(args.save_dir, "{}.p".format(dataname))
        save_pickle(save_path, save_data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--save_dir",
        default=None,
        type=str,
        required=True,
        help="Save Directory",
    )
    parser.add_argument(
        "--src2tgt_model",
        default='transformer.wmt19.en-de.single_model',
        type=str,
        help="torch HUB translation Model(source->target)",
    )
    parser.add_argument(
        "--tgt2src_model",
        default='transformer.wmt19.de-en.single_model',
        type=str,
        help="torch HUB translation Model(target->source)",
    )
    parser.add_argument(
        "--bpe",
        default='fastbpe',
        type=str,
        help="torch HUB translation bpe option",
    )
    parser.add_argument(
        "--tokenizer",
        default='moses',
        type=str,
        help="torch HUB translation tokenizer",
    )
    parser.add_argument(
        "--no_cuda",
        action="store_true",
        help="if you don't want to use CUDA"
    )
    parser.add_argument(
        "--batch_size",
        default=16,
        type=int,
        help="Back-translation Batch size"
    )
    parser.add_argument(
        "--max_len",
        default=300,
        type=int,
        help="Translation Available length"
    )
    parser.add_argument(
        "--temperature",
        default=0.9,
        type=float,
        help="Translation Available length"
    )
    parser.add_argument(
        "--seed",
        default=1,
        type=int,
        help="Translation Available length"
    )
    args = parser.parse_args()

    args.device = 'cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu'
    
    ## make save path
    os.makedirs(args.save_dir, exist_ok=True)

    ## set Seed
    set_seed(args.seed)

    def _print_config(config):
        import pprint
        pp = pprint.PrettyPrinter(indent=4)
        pp.pprint(vars(config))
    _print_config(args)

    run(args)



