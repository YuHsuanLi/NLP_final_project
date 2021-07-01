import smart_open
smart_open.open = smart_open.smart_open
from gensim.models.word2vec import Word2Vec
import gensim.downloader as api
import numpy as np
import torch
import json
import pandas as pd
import string
import pickle as pkl
import re
from transformers import BertTokenizer, BertForSequenceClassification
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, RandomSampler
from torch.nn.utils.rnn import pad_sequence
import os
import random

SEED = 2021
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)

unk_words = []
# TODO CLEAN THE STRING!!!!!!!!!!!!!!!!!
def decontracted(phrase):
    # specific
    phrase = re.sub(r"won't", "will not", phrase)
    phrase = re.sub(r"can\'t", "can not", phrase)

    # general
    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'s", " is", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)

    #tag
    phrase = re.sub(r"@\w+ ", "", phrase)
    return phrase

def remove_handcraft(s, mode='w2v'):
    lst = ['\n', '“', '”']
    for tok in lst:
        s = s.replace(tok, '')

    if mode == 'bert':
        #s = s.replace(',', ' SEP ')
        s = s
    return s

def standard_word(s, mode='w2v'):
    # Turn to lower case
    s = s.lower()
    # Use handcraft rules first
    s = remove_handcraft(s, mode)
    s = decontracted(s)
    # Remove Punctuation
    table = str.maketrans(dict.fromkeys(string.punctuation))
    
    s = s.translate(table)
    return " ".join(s.split()).split(' ')

def word2vec(s):
    global unk_words, model
    word_vectors = [sos_vector]
    for word in s:
        if word in model.wv:
            word_vectors.append(model.wv[word])
        else:
            unk_words.append(word)
            word_vectors.append(unk_vector)
    word_vectors.append(eos_vector)
    return np.array(word_vectors)

def prep_word2token(s):
    # out = ['[CLS]']
    out = []
    for word in s:
        if word == 'SEP':
            out.append('[SEP]')
        else:
            out.append(word)
    #out.append('[SEP]')
    return out
            
def preprocess(s, mode='w2v'):
    s = standard_word(s, mode)
    if mode == 'w2v':
        s = word2vec(s)
    elif mode == 'bert':
        s = prep_word2token(s)
    return s


def get_data_key(MODE, is_dev=False):
    if not is_dev:
        json_path = './public_data_round2/train.json'
        with open(json_path, newline='') as jsonfile:
            raw_data = json.load(jsonfile)
        data_dict = {}
        for data in raw_data:
            idx = data['idx']
            reply_dict = {'text':preprocess(data['reply'],mode=MODE), 'context_idx':data['context_idx'],'reply_text':data['reply']}
            if idx not in data_dict:
                data_dict[idx] = {
                    'idx': idx,
                    'article': preprocess(data['text'], mode=MODE),
                    'label': 0 if data['label'] == 'fake' else 1,
                    'categories': data['categories'],
                    'reply': [reply_dict],
                    'article_text': data['text']
                }
            else:
                data_dict[idx]['reply'].append(reply_dict)
        key_lst = []
        for key in data_dict:
            key_lst.append(key)

        return data_dict, key_lst, raw_data
    else:
        json_path = './public_data_round2/eval.json'
        with open(json_path, newline='') as jsonfile:
            raw_data = json.load(jsonfile)
        data_dict = {}
        for data in raw_data:
            idx = data['idx']
            reply_dict = {'text':preprocess(data['reply'],mode=MODE), 'context_idx':data['context_idx'],'reply_text':data['reply']}
            if idx not in data_dict:
                data_dict[idx] = {
                    'idx': idx,
                    'article': preprocess(data['text'], mode=MODE),
                    'categories': data['categories'],
                    'reply': [reply_dict],
                    'article_text': data['text']
                }
            else:
                data_dict[idx]['reply'].append(reply_dict)
        key_lst = []
        for key in data_dict:
            key_lst.append(key)

        return data_dict, key_lst, raw_data

def load_json_file(filename, train_mode=True):
    json_path = "./"+filename + '.json'
    with open(json_path, newline='') as jsonfile:
        raw_data = json.load(jsonfile)

    data_dict = {}
    for data in raw_data:
        idx = data['idx']
        reply_dict = {'text':preprocess(data['reply'],mode=MODE), 'context_idx':data['context_idx'],'reply_text':data['reply']}
        if idx not in data_dict:
            if train_mode:
                data_dict[idx] = {
                    'idx': idx,
                    'article': preprocess(data['text'], mode=MODE),
                    'label': 0 if data['label'] == 'fake' else 1,
                    'categories': data['categories'],
                    'reply': [reply_dict],
                    'article_text': data['text']
                }
            else:
                data_dict[idx] = {
                    'idx': idx,
                    'article': preprocess(data['text'], mode=MODE),
                    'categories': data['categories'],
                    'reply': [reply_dict],
                    'article_text': data['text']
                }
        else:
            data_dict[idx]['reply'].append(reply_dict)
    
    key_lst = []
    for key in data_dict:
        key_lst.append(key)
    
    return data_dict, key_lst, raw_data

class Dataset(torch.utils.data.Dataset):
    def __init__(self, dict, key_lst, mode='w2v', one_reply=False, is_dev=False):
        self.dict = dict
        self.key_lst = key_lst
        self.mode = mode
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.one_reply = one_reply
        self.is_dev = is_dev
    def __getitem__(self, index):
        idx = self.key_lst[index]
        if not self.is_dev:
            label = self.dict[idx]['label']
        if self.mode == 'w2v':
            article = self.dict[idx]['article']
            replys = []
            for s in self.dict[idx]['reply']:
                replys.append(s['text'])
        elif self.mode == 'bert':
            # Change into string
            article = ' '.join(self.dict[idx]['article'])
            replys = []
            for s in self.dict[idx]['reply']:
                replys.append(' '.join(s['text']))
                
            word_pieces = ["[CLS]"]
            tokens_a = self.tokenizer.tokenize(article)
            word_pieces += tokens_a + ["[SEP]"] 
            len_a = len(word_pieces)
            for reply in replys:
                tokens_r = self.tokenizer.tokenize(reply)
                word_pieces += tokens_r + ["[SEP]"]
                if self.one_reply:
                    break
            len_b = min(len(word_pieces) - len_a, 512- len_a)
            
            ids = self.tokenizer.convert_tokens_to_ids(word_pieces)
            tokens_tensor = torch.tensor(ids)
            if tokens_tensor.shape[0] >512:
                tokens_tensor = tokens_tensor[:512]   
            segments_tensor = torch.tensor([0] * len_a + [1] * len_b, 
                                        dtype=torch.long)
            if not self.is_dev:
                label_tensor = torch.tensor(label)
        if self.is_dev:
            return tokens_tensor, segments_tensor
        return tokens_tensor, segments_tensor, label_tensor 

    def __len__(self):
        return len(self.key_lst)
    

class Dataset_only_article(torch.utils.data.Dataset):
    def __init__(self, dict, key_lst, mode='w2v', one_reply=False, is_dev=False):
        self.dict = dict
        self.key_lst = key_lst
        self.mode = mode
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.one_reply = one_reply
        self.is_dev = is_dev
    def __getitem__(self, index):
        idx = self.key_lst[index]
        if not self.is_dev:
            label = self.dict[idx]['label']
        if self.mode == 'w2v':
            article = self.dict[idx]['article']
            replys = []
            for s in self.dict[idx]['reply']:
                replys.append(s['text'])
        elif self.mode == 'bert':
            # Change into string
            article = ' '.join(self.dict[idx]['article'])
            replys = []
            for s in self.dict[idx]['reply']:
                replys.append(' '.join(s['text']))
                
            word_pieces = ["[CLS]"]
            tokens_a = self.tokenizer.tokenize(article)
            word_pieces += tokens_a + ["[SEP]"] 
            len_a = len(word_pieces)
            len_b = min(len(word_pieces) - len_a, 512- len_a)
            
            ids = self.tokenizer.convert_tokens_to_ids(word_pieces)
            tokens_tensor = torch.tensor(ids)
            if tokens_tensor.shape[0] >512:
                tokens_tensor = tokens_tensor[:512]   
            segments_tensor = torch.tensor([0] * len_a + [1] * len_b, 
                                        dtype=torch.long)
            if not self.is_dev:
                label_tensor = torch.tensor(label)
        if self.is_dev:
            return tokens_tensor, segments_tensor
        return tokens_tensor, segments_tensor, label_tensor 

    def __len__(self):
        return len(self.key_lst)
    
class Dataset_only_replys(torch.utils.data.Dataset):
    def __init__(self, dict, key_lst, mode='w2v', one_reply=False, is_dev=False):
        self.dict = dict
        self.key_lst = key_lst
        self.mode = mode
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.one_reply = one_reply
        self.is_dev = is_dev
    def __getitem__(self, index):
        idx = self.key_lst[index]
        if not self.is_dev:
            label = self.dict[idx]['label']
        if self.mode == 'w2v':
            article = self.dict[idx]['article']
            replys = []
            for s in self.dict[idx]['reply']:
                replys.append(s['text'])
        elif self.mode == 'bert':
            # Change into string
            article = ' '.join(self.dict[idx]['article'])
            replys = []
            for s in self.dict[idx]['reply']:
                replys.append(' '.join(s['text']))
                
            word_pieces = ["[CLS]"]
            #tokens_a = self.tokenizer.tokenize(article)
            #word_pieces += tokens_a + ["[SEP]"] 
            len_a = 0
            for reply in replys:
                tokens_r = self.tokenizer.tokenize(reply)
                word_pieces += tokens_r + ["[SEP]"]
                if self.one_reply:
                    break
            len_b = min(len(word_pieces) - len_a, 512- len_a)
            
            ids = self.tokenizer.convert_tokens_to_ids(word_pieces)
            tokens_tensor = torch.tensor(ids)
            if tokens_tensor.shape[0] >512:
                tokens_tensor = tokens_tensor[:512]   
            segments_tensor = torch.tensor([0] * len_a + [0] * len_b, 
                                        dtype=torch.long)
            if not self.is_dev:
                label_tensor = torch.tensor(label)
        if self.is_dev:
            return tokens_tensor, segments_tensor
        return tokens_tensor, segments_tensor, label_tensor 

    def __len__(self):
        return len(self.key_lst)
    
    
def create_mini_batch(samples):
    tokens_tensors = [s[0] for s in samples]
    segments_tensors = [s[1] for s in samples]
    
    # check if we have labels (we don't have labels in testing phase)
    #if samples[0][2] is not None:
    if len(samples[0]) >= 3:
        label_ids = torch.stack([s[2] for s in samples])
    else:
        label_ids = None
    
    # zero pad 
    tokens_tensors = pad_sequence(tokens_tensors, 
                                  batch_first=True)
    segments_tensors = pad_sequence(segments_tensors, 
                                    batch_first=True)
    
    # attention masks
    masks_tensors = torch.zeros(tokens_tensors.shape, 
                                dtype=torch.long)
    masks_tensors = masks_tensors.masked_fill(
        tokens_tensors != 0, 1)
    
    return tokens_tensors, segments_tensors, masks_tensors, label_ids