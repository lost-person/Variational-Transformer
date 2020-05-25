# coding = utf-8

import torch
import torch.utils.data as data
import random
import math
import os
import logging 
from utils import config
import pickle
from tqdm import tqdm
import numpy as np
import pprint
pp = pprint.PrettyPrinter(indent=1)
import re
import time
import nltk
from collections import deque

class Lang:
    def __init__(self, init_index2word):
        self.word2index = {str(v): int(k) for k, v in init_index2word.items()}
        self.word2count = {str(v): 1 for k, v in init_index2word.items()}
        self.index2word = init_index2word 
        self.n_words = len(init_index2word)  # Count default tokens
      
    def index_words(self, sentence):
        for word in sentence:
            self.index_word(word.strip())

    def index_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

def clean(sentence):
    sentence = sentence.lower()
    sentence = sentence.split()
    return sentence


def load_list_from_path(path):
    data_list = []
    with open(path, 'r') as f:
        for line in f:
            data_list.append(line.strip())
    return data_list


def read_langs(vocab: Lang):
    data_train = {'context':[], 'target':[]}
    data_dev = {'context':[], 'target':[]}
    data_test = {'context':[], 'target':[]}
    
    with open('./data/ubuntu/vocab.ori', 'r', encoding='utf-8') as f:
        for word in f:
            vocab.index_word(word.strip())

    train_context = load_list_from_path('./data/ubuntu/train/context.txt')
    train_target = load_list_from_path('./data/ubuntu/train/response.txt')

    dev_context = load_list_from_path('./data/ubuntu/dev/context.txt')
    dev_target = load_list_from_path('./data/ubuntu/dev/response.txt')
    
    test_context = load_list_from_path('./data/ubuntu/test/context.txt')
    test_target = load_list_from_path('./data/ubuntu/test/response.txt')

    for context in train_context:
        u_list = []
        for u in context.split(" <eou> "):
            u_list.append(u.split(' '))
        data_train['context'].append(u_list)

    for target in train_target:
        data_train['target'].append(target.split(' '))
    
    for context in dev_context:
        u_list = []
        for u in context.split(" <eou> "):
            u_list.append(u.split(' '))
        data_dev['context'].append(u_list)

    for target in dev_target:
        data_dev['target'].append(target.split(' '))
    
    for context in test_context:
        u_list = []
        for u in context.split(" <eou> "):
            u_list.append(u.split(' '))
        data_test['context'].append(u_list)

    for target in test_target:
        data_test['target'].append(target.split(' '))

    return data_train, data_dev, data_test, vocab

def load_dataset():
    if(os.path.exists('./data/ubuntu/dataset_ubuntu.p')):
        print("LOADING ubuntu")
        with open('./data/ubuntu/dataset_ubuntu.p', "rb") as f:
            [data_tra, data_val, data_tst, vocab] = pickle.load(f)
    else:
        print("Building dataset...")
        data_tra, data_val, data_tst, vocab = read_langs(vocab=Lang({config.UNK_idx: "UNK", config.PAD_idx: "PAD", config.EOS_idx: "EOS", config.SOS_idx: "SOS"}))
        with open('./data/ubuntu/dataset_ubuntu.p', "wb") as f:
            pickle.dump([data_tra, data_val, data_tst, vocab], f)
            print("Saved PICKLE")
    for i in range(3):
        print('[context]:', [' '.join(u) for u in data_tra['context'][i]])
        print('[target]:', ' '.join(data_tra['target'][i]))
        print(" ")
    return data_tra, data_val, data_tst, vocab
