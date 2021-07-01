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
import dataloader
import train
import random

SEED = 2021
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)

is_dev = True

# Set parameters
BATCH_SIZE = 8

output_PATH = '/eva_data_2/yu_hsuan_li/NLP/output/round2_own_classifier_only_article_epoch_20'
if is_dev == False: 
    if(os.path.isdir(output_PATH)==False):
        os.mkdir(output_PATH)
device = torch.device('cuda:6' if torch.cuda.is_available() else 'cpu')    
MODE = 'bert'

# copy files
'''
backup_dir = os.path.join(output_PATH, 'backup_files')
os.makedirs(backup_dir, exist_ok=True)
os.system('cp *.py %s/' % backup_dir)
'''
#os.system('cp *.ipynb %s/' % backup_dir)

# dataloader
#train
'''
data_dict, key_lst, raw_data = dataloader.get_data_key(MODE, is_dev=False)
dataset = dataloader.Dataset(data_dict, key_lst, mode=MODE, one_reply=False, is_dev=False, device = device)
trainloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
'''
#test
data_dict, key_lst, raw_data = dataloader.get_data_key(MODE, is_dev=True)
dataset = dataloader.Dataset(data_dict, key_lst, mode=MODE, one_reply=False, is_dev=True, device = device)
testloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

model = torch.nn.Linear(768, 2).to(device)
#model = torch.nn.Linear(1536, 2).to(device)
model = model = torch.load('/eva_data_2/yu_hsuan_li/NLP/output/round2_own_classifier_only_article_epoch_20/model_9.pth', map_location=device).to(device)
optimizer = optim.Adam(model.parameters(), lr=2e-5)

if is_dev:
    train.eva_and_csv(model, raw_data, testloader, device = device)
else:
    for epoch in range(20):
        print(epoch)
        model = train.train(model=model, optimizer=optimizer, train_loader=trainloader, epoch = epoch, output_PATH = output_PATH, device=device)
        acc, cm = train.test(model, trainloader, device = device)
        print(cm)
        print('test_acc', str(acc))
    train.eva_and_csv(model, raw_data, testloader, device = device)