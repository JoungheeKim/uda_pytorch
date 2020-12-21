# UDA Pytorch
This is pytorch implementation project of [**Unsupervised Data Augmentation for consistency Training, NeurIPS 2020**](https://arxiv.org/abs/1904.12848)


## Overview



## Reference
- [[BLOG]](https://nlp.stanford.edu/blog/maximum-likelihood-decoding-with-rnns-the-good-the-bad-and-the-ugly/#:~:text=Temperature%20sampling%20is%20a%20standard,semantic%20distortions%20in%20the%20process.) Maximum Likelihood Decoding with RNNs - the good, the bad, and the ugly
- [[PAPER]](https://arxiv.org/abs/1904.12848) Unsupervised Data Augmentation for Consistency Training, Qizhe at el, NeurIPS 2020
- [[GITHUB]](https://github.com/google-research/uda) Unsupervised Data Augmentation Tensorflow Implementation

## Acknowledgements
 - [`train_mlm.py`](https://github.com/JoungheeKim/uda_pytorch/blob/main/src/train_mlm.py) : This implementation uses code  from folling repos [huggingface `run_mlm.py`](https://github.com/huggingface/transformers/blob/master/examples/language-modeling/run_mlm.py) to pretrain IMDB datasets.
 - [`train.py`](https://github.com/JoungheeKim/uda_pytorch/blob/main/src/train.py) : This is inspired by [JungHoon Lee Code Style](https://github.com/JhnLee)
