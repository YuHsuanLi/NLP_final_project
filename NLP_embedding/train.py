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
import csv

SEED = 2021
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)

def confusion_matrix(gt_list, pred_list):
    from sklearn.metrics import confusion_matrix
    confusion_mat = confusion_matrix(np.concatenate(gt_list,axis=0),np.concatenate(pred_list,axis=0))
    return confusion_mat

def test(model,
          test_loader,
          device = None,
          return_pred_list = False):
    
    # initialize running values
    model.eval()
    correct = 0
    total = 0
    pred_list = []
    gt_list = []
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            #print(i)
            tokens_tensors, labels= data[0], data[1]
            tokens_tensors, labels = tokens_tensors.to(device), labels.to(device)
            outputs = model(tokens_tensors)        
            pred = torch.argmax(outputs, 1)         
            #labels = labels.cpu().detach().numpy()
            pred_list += [pred.cpu().detach().numpy()]
            gt_list += [labels.cpu().detach().numpy()]            
            total += labels.size(0)          
            labels = labels.reshape(-1)
            correct += (pred == labels).sum().item()

            
    if return_pred_list:
        return correct/total, confusion_matrix(gt_list, pred_list), pred_list
    return correct/total, confusion_matrix(gt_list, pred_list)


def eva_and_csv(model,
              dev_data, 
              test_loader,
              device = None):
    
    # initialize running values
    model.eval()
    correct = 0
    total = 0
    pred_list = []
    gt_list = []
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            #print(i)
            tokens_tensors = data
            tokens_tensors = tokens_tensors.to(device)
            outputs = model(tokens_tensors)        
            pred = torch.argmax(outputs, 1)
            pred_list += [pred.cpu().detach().numpy()]

    pred_list_np = np.concatenate(pred_list)
    
    with open('round2_own_classifier_only_article_epoch_10_eval.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['idx', 'context_idx', 'label'])
        data_dict={}
        label_idx = 0
        for data in dev_data:
            idx = data['idx']
            if idx not in data_dict:
                data_dict[idx] = {
                    'idx': idx,
                    'label': pred_list_np[label_idx]
                }
                label_idx += 1
            writer.writerow([idx, data['context_idx'], 'real' if data_dict[idx]['label'] else 'fake'])


def train(model,
          optimizer,
          train_loader,
          criterion = nn.CrossEntropyLoss(),          
          epoch = 0 ,
          best_valid_loss = float("Inf"),
          output_PATH = None, 
          device = None):
    
    # initialize running values
    running_loss = 0.0
    valid_running_loss = 0.0
    global_step = 0
    train_loss_list = []
    valid_loss_list = []
    global_steps_list = []

    # training loop
    model.train()
    #for epoch in range(num_epochs):
    correct = 0
    total = 0
    for i, data in enumerate(train_loader):
        tokens_tensors, labels= data[0], data[1]
        tokens_tensors, labels = tokens_tensors.to(device), labels.to(device)
        outputs = model(tokens_tensors)        
        loss = criterion(outputs, labels)
        pred = torch.argmax(outputs, 1)
        total += labels.size(0)
        labels = labels.reshape(-1)
        correct += (pred == labels).sum().item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # update running values
        running_loss += loss.item()
        global_step += 1
        #print(running_loss/global_step)
    torch.save(model, output_PATH+'/model_' + str(epoch) +'.pth')
    print('running loss:' + str(running_loss/global_step))
    print('train acc:', str(correct/total))
    return model