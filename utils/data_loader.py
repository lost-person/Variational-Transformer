
import torch
import torch.utils.data as data
import random
import math
import os
import logging 
from utils import config
import pickle
from tqdm import tqdm
import pprint
pp = pprint.PrettyPrinter(indent=1)
import re
import time
from model.common_layer import write_config
if config.dataset=="empathetic":
    from utils.persona_ed_reader import load_dataset
elif config.dataset=='mojitalk':
    from utils.mojitalk_reader import load_dataset
elif config.dataset=='cornell':
    from utils.cornell_reader import load_dataset
elif config.dataset=='ubuntu':
    from utils.ubuntu_reader import load_dataset

class Dataset(data.Dataset):
    """Custom data.Dataset compatible with data.DataLoader."""
    def __init__(self, data, vocab):
        """Reads source and target sequences from txt files."""
        self.vocab = vocab
        self.data = data
        if config.dataset=="empathetic":
            self.emo_map = {'surprised': 9, 'excited': 10, 'annoyed': 11, 'proud': 12, 'angry': 13, 'sad': 14, 'grateful': 15, 'lonely': 16, 'impressed': 17, 'afraid': 18, 'disgusted': 19, 'confident': 20, 'terrified': 21, 'hopeful': 22, 'anxious': 23, 'disappointed': 24, 'joyful': 25, 'prepared': 26, 'guilty': 27, 'furious': 28, 'nostalgic': 29, 'jealous': 30, 'anticipating': 31, 'embarrassed': 32, 'content': 33, 'devastated': 34, 'sentimental': 35, 'caring': 36, 'trusting': 37, 'ashamed': 38, 'apprehensive': 39, 'faithful': 40}
        # else:
        elif config.dataset=='mojitalk':
            self.emo_map = {'😁': 9, '😂': 10, '😄': 11, '😅': 12, '😉': 13, '😊': 14, '😋': 15, '😎': 16, '😍': 17, '😘': 18, '☺': 19, '😐': 20, '😑': 21, '😏': 22, '😣': 23, '😪': 24, '😫': 25, '😴': 26, '😌': 27, '😜': 28, '😒': 29, '😓': 30, '😔': 31, '😕': 32, '😖': 33, '😞': 34, '😤': 35, '😢': 36, '😭': 37, '😩': 38, '😬': 39, '😳': 40, '😡': 41, '😠': 42, '😷': 43, '😈': 44, '💀': 45, '🙈': 46, '🙊': 47, '🙅': 48, '💁': 49, '💪': 50, '✌': 51, '✋': 52, '👌': 53, '👍': 54, '👊': 55, '👏': 56, '🙌': 57, '🙏': 58, '👀': 59, '❤': 60, '💔': 61, '💕': 62, '💖': 63, '💙': 64, '💜': 65, '💟': 66, '✨': 67, '♥': 68, '🎶': 69, '🎧': 70, '🔫': 71, '💯': 72}
        else:
            self.emo_map = {}
    def __len__(self):
        return len(self.data["target"])

    def __getitem__(self, index):
        """Returns one data pair (source and target)."""
        item = {}
        item["context_text"] = self.data["context"][index]
        item["target_text"] = self.data["target"][index]
        if config.dataset in ["empathetic", "mojitalk"]:
            item["emotion_text"] = self.data["emotion"][index]
            item["emotion"], item["emotion_label"] = self.preprocess_emo(item["emotion_text"], self.emo_map)
        if config.dataset=="empathetic":
            item["context"], item["context_mask"] = self.preprocess(item["context_text"])
            item["posterior"], item["posterior_mask"] = self.preprocess(arr=[item["target_text"]], posterior=True)
        elif config.dataset in ['cornell', 'ubuntu']:
            item["context"], item["context_mask"] = self.preprocess(item["context_text"])
            item["posterior"], item["posterior_mask"] = self.preprocess(arr=[item["target_text"]], posterior=True)
        else:
            item["context"], item["context_mask"] = self.preprocess(arr=[item["context_text"]],meta=item["emotion_label"])
            item["posterior"], item["posterior_mask"] = self.preprocess(arr=[item["target_text"]], meta=item["emotion_label"],posterior=True)
        item["target"] = self.preprocess(item["target_text"], anw=True)
        return item


    def preprocess(self, arr, anw=False, meta=None, posterior=False):
        """Converts words to ids."""
        if(anw):
            sequence = [self.vocab.word2index[word] if word in self.vocab.word2index else config.UNK_idx for word in arr] + [config.EOS_idx]
            if config.dataset in ['cornell', 'ubuntu']:
                sequence = sequence[:config.max_seq_len - 1] + [config.EOS_idx]
            return torch.LongTensor(sequence)
        else:
            X_dial = [config.CLS1_idx,meta] if posterior else [config.CLS_idx,meta]
            X_mask = [config.CLS1_idx,meta] if posterior else [config.CLS_idx,meta]
            if config.dataset=="empathetic":
                X_dial = [config.CLS1_idx] if posterior else [config.CLS_idx]
                X_mask = [config.CLS1_idx] if posterior else [config.CLS_idx]
            if (config.model=="seq2seq" or config.model=="cvae" or config.dataset in ['cornell', 'ubuntu']):
                X_dial = []
                X_mask = []
            for i, sentence in enumerate(arr):
                if config.dataset in ['cornell', 'ubuntu']:
                    if not posterior:
                        X_id = [self.vocab.word2index[word] if word in self.vocab.word2index else config.UNK_idx for word in sentence]
                        X_id = X_id[:config.max_seq_len - 1] + [config.EOS_idx]
                        X_dial.append(X_id + [config.PAD_idx] * (config.max_seq_len - len(X_id)))
                        X_mask.append([4] * config.max_seq_len ) # 4 -> User Id
                    else:
                        X_dial += [self.vocab.word2index[word] if word in self.vocab.word2index else config.UNK_idx for word in sentence] 
                        if config.dataset in ['cornell', 'ubuntu']:
                            X_dial = X_dial[:config.max_seq_len - 1] + [config.EOS_idx]
                            X_mask += [4 for _ in range(len(X_dial))]
                        else:
                            X_mask += [4 for _ in range(len(sentence))] # 4 -> User Id
                else:
                    X_dial += [self.vocab.word2index[word] if word in self.vocab.word2index else config.UNK_idx for word in sentence]
                    spk = self.vocab.word2index["USR"] if i % 2 == 0 else self.vocab.word2index["SYS"]
                    if posterior: spk = self.vocab.word2index["SYS"]
                    X_mask += [spk for _ in range(len(sentence))]
            
            if config.dataset in ['cornell', 'ubuntu'] and not posterior:
                X_dial, X_mask = X_dial[:config.max_conv_len], X_mask[:config.max_conv_len]
            assert len(X_dial) == len(X_mask)

            return torch.LongTensor(X_dial), torch.LongTensor(X_mask)
    

    # def preprocess_mojitalk(self, arr, anw=False, meta=None, ):
    #     """Converts words to ids."""
    #     if(anw):
    #         sequence = [self.vocab.word2index[word] if word in self.vocab.word2index else config.UNK_idx for word in arr] + [config.EOS_idx]
    #         return torch.LongTensor(sequence)
    #     else:

    #         X_dial = [config.CLS_idx] + [self.vocab.word2index[word] if word in self.vocab.word2index else config.UNK_idx for word in arr]
    #         return torch.LongTensor(X_dial)

    def preprocess_emo(self, emotion, emo_map):
        # program = [0]*len(emo_map)
        # program[emo_map[emotion]] = 1
        return 0, emo_map[emotion]

def collate_fn(data):
    def merge(sequences, mode=None):
        lengths = [len(seq) for seq in sequences]
        if config.dataset in ['cornell', 'ubuntu'] and mode == 'context':
            padded_seqs = torch.cat(sequences, dim=0)
        else:
            padded_seqs = torch.ones(len(sequences), max(lengths)).long() ## padding index 1
            for i, seq in enumerate(sequences):
                end = lengths[i]
                padded_seqs[i, :end] = seq[:end]
        return padded_seqs, torch.LongTensor(lengths) 


    data.sort(key=lambda x: len(x["context"]), reverse=True) ## sort by source seq
    item_info = {}
    for key in data[0].keys():
        item_info[key] = [d[key] for d in data]

    ## input
    input_batch, input_lengths     = merge(item_info['context'], mode='context')
    posterior_batch, posterior_lengths     = merge(item_info['posterior'])
    input_mask, input_mask_lengths = merge(item_info['context_mask'], mode='context')
    posterior_mask, posterior_mask_lengths = merge(item_info['posterior_mask'])
    ## Target
    target_batch, target_lengths   = merge(item_info['target'])


    if config.USE_CUDA:
        input_batch = input_batch.cuda()
        input_lengths = input_lengths.cuda()
        posterior_batch = posterior_batch.cuda()
        posterior_lengths = posterior_lengths.cuda()
        posterior_mask = posterior_mask.cuda()
        input_mask = input_mask.cuda()
        input_mask_lengths = input_mask_lengths.cuda()
        target_batch = target_batch.cuda()
        target_lengths = target_lengths.cuda()
 
    d = {}
    d["input_batch"] = input_batch
    d["input_lengths"] = input_lengths
    d["input_mask"] = input_mask
    d["posterior_batch"] = posterior_batch
    d["posterior_lengths"] = posterior_lengths
    d["posterior_mask"] = posterior_mask
    d["target_batch"] = target_batch
    d["target_lengths"] = target_lengths
    ##program
    if config.dataset in ["empathetic", "mojitalk"]:
        d["target_program"] = item_info['emotion']
        d["program_label"] = torch.LongTensor(item_info['emotion_label'])
        if config.USE_CUDA:
            d["program_label"] = d["program_label"].cuda()
    ##text
    d["input_txt"] = item_info['context_text']
    d["target_txt"] = item_info['target_text']
    if config.dataset not in ['cornell', 'ubuntu']:
        d["program_txt"] = item_info['emotion_text']
    return d 


def prepare_data_seq(batch_size=32):  

    pairs_tra, pairs_val, pairs_tst, vocab = load_dataset()

    logging.info("Vocab  {} ".format(vocab.n_words))

    dataset_train = Dataset(pairs_tra, vocab)
    data_loader_tra = torch.utils.data.DataLoader(dataset=dataset_train,
                                                 batch_size=batch_size,
                                                 shuffle=True, collate_fn=collate_fn,
                                                 drop_last=True)

    dataset_valid = Dataset(pairs_val, vocab)
    data_loader_val = torch.utils.data.DataLoader(dataset=dataset_valid,
                                                 batch_size=batch_size,
                                                 shuffle=True, collate_fn=collate_fn,
                                                 drop_last=True)
    #print('val len:',len(dataset_valid))
    dataset_test = Dataset(pairs_tst, vocab)
    data_loader_tst = torch.utils.data.DataLoader(dataset=dataset_test,
                                                 batch_size=1,
                                                 shuffle=False, collate_fn=collate_fn,
                                                 drop_last=True)
    write_config()
    return data_loader_tra, data_loader_val, data_loader_tst, vocab, len(dataset_train.emo_map)