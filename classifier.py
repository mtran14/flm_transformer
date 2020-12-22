import torch
from transformer.nn_transformer import TRANSFORMER
from downstream.model import example_classifier, RnnClassifier, FeedForwardClassifier
from downstream.solver import get_optimizer
from downstream.dataloader_ds import SchizophreniaDataset, SchizophreniaSegmentDataset
import pandas as pd
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader
import re
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.linear_model import LogisticRegression
import torch.nn as nn
import sys

seed = 123
device = 'cuda' if torch.cuda.is_available() else 'cpu'

config = {
            'mode'     : 'classification',
            'sample_rate' : 1,
            'hidden_size'       : 128,
            'pre_linear_dims'       : [20], 'post_linear_dims': [20],'drop':0.2,
            'concat': 1, 'layers': 3, 'linear': False,
        }

#mymodel = FeedForwardClassifier(136, 2, config)
#mytensor = torch.from_numpy(np.random.rand(16,50,136))
#labels = torch.zeros(16,dtype=torch.long)
#label_mask = (mytensor.sum(dim=-1) != 0).type(torch.LongTensor).to(device='cpu', dtype=torch.long)
#valid_lengths = label_mask.sum(dim=1)
#loss, result, correct, valid = mymodel.forward(mytensor.float(), labels, valid_lengths)

def dict_acc(input_dict):
    cnt = 0
    cor = 0
    for key in input_dict.keys():
        if(sum(input_dict[key])/len(input_dict[key]) >= 0.5):
            cor += 1
        cnt += 1
    return cor/cnt

def duplicate(features, n_times, d_bool=False):
    if(d_bool):
        return torch.cat((features.reshape(-1,1),)*n_times, dim=1)
    return features


torch.manual_seed(seed)

subset = sys.argv[1]
n_fold = 5
sets = ["data/train-clean-schz_chunk_0.csv","data/train-clean-schz_chunk_1.csv",\
        "data/train-clean-schz_chunk_2.csv","data/train-clean-schz_chunk_3.csv"]
tables = [pd.read_csv(s, header=None) for s in sets]
table = pd.concat(tables, ignore_index=True).values

table_filter = []
for row in table:
    if(subset in row[0]):
        table_filter.append(row)
table = np.array(table_filter)

kf = KFold(n_splits=n_fold, shuffle=True, random_state=seed)
n_step = 200
n_val = 100
segment_size = 500
bs = 12
val_every = 40

pretrain = True

overall_w = []
overall_f = []
for train_index, test_index in kf.split(table):
    train_files = table[train_index[:-n_val]][:,0]
    train_labels = []
    val_files = table[train_index[-n_val:]][:,0]
    dev_labels = []
    for file in train_files:
        current_label = 0 if (int(re.search(r'\d{4}', file)[0]) >= 8000) else 1
        train_labels.append(current_label)
    for file in val_files:
        current_label = 0 if (int(re.search(r'\d{4}', file)[0]) >= 8000) else 1
        dev_labels.append(current_label)    
        
    #train_dataset = SchizophreniaDataset(train_files, segment_size, train_labels)
    train_dataset = SchizophreniaSegmentDataset(train_files, train_labels, max_len=segment_size)
    train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True)
    
    #dev_dataset = SchizophreniaDataset(val_files, segment_size, dev_labels)
    dev_dataset = SchizophreniaSegmentDataset(val_files, dev_labels, max_len=segment_size)
    dev_loader = DataLoader(dev_dataset, batch_size=bs, shuffle=True)    
    ###########################
    #load transformer if pretrained
    if(pretrain):
        options = {
            'ckpt_file'     : sys.argv[2],
            'load_pretrain' : 'True',
            'no_grad'       : 'True',
            'dropout'       : 'default',
            'spec_aug'      : 'False',
            'spec_aug_prev' : 'True',
            'weighted_sum'  : 'False',
            'select_layer'  : -1,
            'permute_input' : 'False',
        }
        transformer = TRANSFORMER(options=options, inp_dim=0) # set `inpu_dim=0` to auto load the `inp_dim` from `ckpt_file`
        
        # setup your downstream class model
        classifier = RnnClassifier(272, 2, config).to(device)
        # construct the optimizer
        params = list(list(transformer.named_parameters()) + list(classifier.named_parameters()))
        optimizer = get_optimizer(params=params, lr=4e-3, warmup_proportion=0.7, training_steps=30000)
            
    ###########################
    else:
        #init model and optimizer
        #classifier = example_classifier(input_dim=136, hidden_dim=64, class_num=2).to(device)
        classifier = RnnClassifier(136, 2, config).to(device)
        params = list(list(classifier.named_parameters()))
        optimizer = get_optimizer(params=params, lr=4e-3, warmup_proportion=0.7, training_steps=10000) 
    ###########################

    if(pretrain):
        transformer.eval()
    train_set = None
    test_set = None
    dev_set = None  
    train_label = []
    dev_label = []
    test_label = []
    for _ in range(n_step):
        
        for k, batch in enumerate(train_loader):
            batch_data, batch_labels, file_names = batch
            batch_data = batch_data.to(device)
            batch_labels = duplicate(batch_labels, segment_size, d_bool=False).to(device)
            
            if(pretrain):
                reps = transformer(batch_data)
                batch_data = reps
            
            #current_data = np.array(batch_data.detach().cpu())
            #current_data = current_data.reshape(current_data.shape[0] * current_data.shape[1], current_data.shape[2])
            #try:
                #train_set = np.concatenate((train_set, current_data), axis=0)
            #except:
                #train_set = current_data
            #train_label += list(np.array(torch.cat((batch_labels,)*segment_size, dim=0).cpu()))
            
            label_mask = (batch_data.sum(dim=-1) != 0).type(torch.LongTensor).to(device=device, dtype=torch.long)
            valid_lengths = label_mask.sum(dim=1)        
            
            optimizer.zero_grad()
            loss, result, correct, valid = classifier.forward(batch_data.float(), batch_labels, valid_lengths)
            loss.backward()
            optimizer.step()  
            
        if(_ % val_every == 0):   
            classifier.eval()
            if(pretrain):
                transformer.eval()
            fold_acc_window = []
            #fold_acc_file = {}
            with torch.no_grad():
                for _, batch in enumerate(dev_loader):
                    batch_data, batch_labels, file_names = batch
                    batch_data = batch_data.to(device)
                    batch_labels = duplicate(batch_labels, segment_size, d_bool=False).to(device) 
            
                    if(pretrain):
                        reps = transformer(batch_data)
                        batch_data = reps
                        
                    #current_data = np.array(batch_data.detach().cpu())
                    #current_data = current_data.reshape(current_data.shape[0] * current_data.shape[1], current_data.shape[2])
                    #try:
                        #dev_set = np.concatenate((dev_set, current_data), axis=0)
                    #except:
                        #dev_set = current_data     
                    #dev_label += list(np.array(torch.cat((batch_labels,)*segment_size, dim=0).cpu()))
                    label_mask = (batch_data.sum(dim=-1) != 0).type(torch.LongTensor).to(device=device, dtype=torch.long)
                    valid_lengths = label_mask.sum(dim=1)          
            
                    
                    loss, result, correct, valid = classifier.forward(batch_data.float(), batch_labels, valid_lengths)
                    preds = result.argmax(dim=1).detach().cpu().numpy()
                    batch_labels = batch_labels.detach().cpu().numpy()
                    batch_acc = accuracy_score(batch_labels, preds)
                    fold_acc_window.append(correct.item()/valid.item())
                    
                    #for i in range(batch_labels.shape[0]):
                        #current_file = file_names[i]
                        #correct_bool = 1 if preds[i]==batch_labels[i] else 0
                        #try:
                            #fold_acc_file[current_file].append(correct_bool)
                        #except:
                            #fold_acc_file[current_file] = [correct_bool]  
                            
            test_files = table[test_index][:,0]
            test_labels = []
            for file in test_files:
                current_label = 0 if (int(re.search(r'\d{4}', file)[0]) >= 8000) else 1
                test_labels.append(current_label)
                            
            
            test_dataset = SchizophreniaSegmentDataset(test_files, test_labels, max_len=segment_size)
            test_loader = DataLoader(test_dataset, batch_size=bs, shuffle=True)  
            
            
            fold_acc_window_test = []
            fold_acc_file_test = {}
            with torch.no_grad():
                for _, batch in enumerate(test_loader):
                    batch_data, batch_labels, file_names = batch
                    batch_data = batch_data.to(device)
                    batch_labels = duplicate(batch_labels, segment_size, d_bool=False).to(device) 
                    
                    if(pretrain):
                        reps = transformer(batch_data)
                        batch_data = reps
                        
                    #current_data = np.array(batch_data.detach().cpu())
                    #current_data = current_data.reshape(current_data.shape[0] * current_data.shape[1], current_data.shape[2])
                    #try:
                        #test_set = np.concatenate((test_set, current_data), axis=0)
                    #except:
                        #test_set = current_data                             
                    #test_label += list(np.array(torch.cat((batch_labels,)*segment_size, dim=0).cpu()))
                    label_mask = (batch_data.sum(dim=-1) != 0).type(torch.LongTensor).to(device=device, dtype=torch.long)
                    valid_lengths = label_mask.sum(dim=1)          
                    
                    #loss, result = classifier(batch_data.float(), batch_labels)
                    loss, result, correct, valid = classifier.forward(batch_data.float(), batch_labels, valid_lengths)
                    preds = result.argmax(dim=1).detach().cpu().numpy()
                    batch_labels = batch_labels.detach().cpu().numpy()
                    batch_acc = accuracy_score(batch_labels, preds)
                    fold_acc_window_test.append(correct.item()/valid.item())
                    
                    #for i in range(batch_labels.shape[0]):
                        #current_file = file_names[i]
                        #correct_bool = 1 if preds[i]==batch_labels[i] else 0
                        #try:
                            #fold_acc_file_test[current_file].append(correct_bool)
                        #except:
                            #fold_acc_file_test[current_file] = [correct_bool]
            #print(np.mean(fold_acc_window_test), dict_acc(fold_acc_file_test))        
            #print("Epoch ", e, np.mean(epoch_loss), \
                  #"Dev Set: ", np.mean(fold_acc_window), dict_acc(fold_acc_file), \
                  #"Test Set: ", np.mean(fold_acc_window_test), dict_acc(fold_acc_file_test))
            classifier.train()
            if(pretrain):
                transformer.train()
            print("Dev: ", np.mean(fold_acc_window), "Test: ", np.mean(fold_acc_window_test), \
                  " P(1):", 1-sum(test_labels)/len(test_labels), " P(0):", sum(test_labels)/len(test_labels))
        
    #test_files = table[test_index][:,0]
    #test_labels = []
    #for file in test_files:
        #current_label = 0 if (int(re.search(r'\d{4}', file)[0]) >= 8000) else 1
        #test_labels.append(current_label)
        
    #test_dataset = SchizophreniaDataset(test_files, 100, test_labels)
    #test_loader = DataLoader(test_dataset, batch_size=16, shuffle=True)  
    
    #classifier.eval()
    #if(pretrain):
        #transformer.eval()
    #fold_acc_window = []
    #fold_acc_file = {}
    #for _, batch in enumerate(test_loader):
        #batch_data, batch_labels, file_names = batch
        #batch_data = batch_data.to(device)
        #batch_labels = batch_labels.to(device) 
        
        #if(pretrain):
            #reps = transformer(batch_data)
            #batch_data = reps
            
        #label_mask = (batch_data.sum(dim=-1) != 0).type(torch.LongTensor).to(device=device, dtype=torch.long)
        #valid_lengths = label_mask.sum(dim=1)          
        
        ##loss, result = classifier(batch_data.float(), batch_labels)
        #loss, result, correct, valid = classifier.forward(batch_data.float(), batch_labels, valid_lengths)
        #preds = result.argmax(dim=1).detach().cpu().numpy()
        #batch_labels = batch_labels.detach().cpu().numpy()
        #batch_acc = accuracy_score(batch_labels, preds)
        #fold_acc_window.append(batch_acc)
        
        #for i in range(batch_labels.shape[0]):
            #current_file = file_names[i]
            #correct_bool = 1 if preds[i]==batch_labels[i] else 0
            #try:
                #fold_acc_file[current_file].append(correct_bool)
            #except:
                #fold_acc_file[current_file] = [correct_bool]
    #print(np.mean(fold_acc_window), dict_acc(fold_acc_file))
    #overall_w.append(np.mean(fold_acc_window))
    #overall_f.append(dict_acc(fold_acc_file))
    
    
#print("Total: ", np.mean(overall_w), np.mean(overall_f))

#123 -> 0.7283333333333333