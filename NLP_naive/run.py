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

is_dev = False

# Set parameters
BATCH_SIZE = 8

output_PATH = '/eva_data_2/yu_hsuan_li/NLP/output/round2_article_and_one_reply_epoch_10'
if is_dev == False: 
    if(os.path.isdir(output_PATH)==False):
        os.mkdir(output_PATH)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')    
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
data_dict, key_lst, raw_data = dataloader.get_data_key(MODE, is_dev=False)
dataset = dataloader.Dataset(data_dict, key_lst, mode=MODE, one_reply=True, is_dev=False)
trainloader = DataLoader(dataset, batch_size=BATCH_SIZE, collate_fn=dataloader.create_mini_batch, shuffle=True)
#test
data_dict, key_lst, raw_data = dataloader.get_data_key(MODE, is_dev=True)
dataset = dataloader.Dataset(data_dict, key_lst, mode=MODE, one_reply=True, is_dev=True)
testloader = DataLoader(dataset, batch_size=BATCH_SIZE, collate_fn=dataloader.create_mini_batch, shuffle=False)

#train_data, test_data = torch.utils.data.random_split(dataset, [31516, 3502])
#trainloader = DataLoader(train_data, batch_size=BATCH_SIZE, collate_fn=dataloader.create_mini_batch, shuffle=True)
#testloader = DataLoader(test_data, batch_size=BATCH_SIZE, collate_fn=dataloader.create_mini_batch, shuffle=False)
'''
if is_dev:
    testloader = DataLoader(dataset, batch_size=BATCH_SIZE, collate_fn=dataloader.create_mini_batch, shuffle=False)
else:
    trainloader = DataLoader(dataset, batch_size=BATCH_SIZE, collate_fn=dataloader.create_mini_batch, shuffle=True)
'''    
    
# set model
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2).to(device) #try "bert-base-cased"
optimizer = optim.Adam(model.parameters(), lr=2e-5)
#model = torch.load('/eva_data_2/yu_hsuan_li/NLP/output/round2_only_replys_epoch_10/model_9.pth', map_location=device).to(device)
if is_dev:
    train.eva_and_csv(model, raw_data, testloader, device = device)
else:
    for epoch in range(10):
        print(epoch)
        model = train.train(model=model, optimizer=optimizer, train_loader=trainloader, epoch = epoch, output_PATH = output_PATH, device=device)
        acc, cm = train.test(model, trainloader, device = device)
        print(cm)
        print('test_acc', str(acc))
    train.eva_and_csv(model, raw_data, testloader, device = device)