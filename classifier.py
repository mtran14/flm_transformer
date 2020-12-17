import torch
from transformer.nn_transformer import TRANSFORMER
from downstream.model import example_classifier
from downstream.solver import get_optimizer
from downstream.dataloader_ds import SchizophreniaDataset
import pandas as pd
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader
import re
from sklearn.metrics import accuracy_score
import numpy as np

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def dict_acc(input_dict):
    cnt = 0
    cor = 0
    for key in input_dict.keys():
        if(sum(input_dict[key])/len(input_dict[key]) >= 0.5):
            cor += 1
        cnt += 1
    return cor/cnt

n_fold = 10
sets = ["data/train-clean-360_chunk_0.csv","data/train-clean-360_chunk_1.csv"]
tables = [pd.read_csv(s, header=None) for s in sets]
table = pd.concat(tables, ignore_index=True).values
kf = KFold(n_splits=n_fold, shuffle=True, random_state=1)

pretrain = True

overall_w = []
overall_f = []
for train_index, test_index in kf.split(table):
    train_files = table[train_index][:,0]
    train_labels = []
    for file in train_files:
        current_label = 0 if (int(re.search(r'\d{4}', file)[0]) >= 8000) else 1
        train_labels.append(current_label)
    train_dataset = SchizophreniaDataset(train_files, 100, train_labels)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    ###########################
    #load transformer if pretrained
    if(pretrain):
        options = {
            'ckpt_file'     : './result/result_transformer/mockingjay_fbankBase/states-500000.ckpt',
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
        classifier = example_classifier(input_dim=768, hidden_dim=64, class_num=2).to(device)
        
        # construct the optimizer
        params = list(list(classifier.named_parameters()))
        optimizer = get_optimizer(params=params, lr=4e-3, warmup_proportion=0.7, training_steps=50000)
            
    ###########################
    else:
        #init model and optimizer
        classifier = example_classifier(input_dim=136, hidden_dim=64, class_num=2).to(device)
        params = list(list(classifier.named_parameters()))
        optimizer = get_optimizer(params=params, lr=4e-3, warmup_proportion=0.7, training_steps=50000)    
    ###########################
    for _, batch in enumerate(train_loader):
        batch_data, batch_labels, file_names = batch
        batch_data = batch_data.to(device)
        batch_labels = batch_labels.to(device)
        
        if(pretrain):
            reps = transformer(batch_data)
            batch_data = reps
        
        optimizer.zero_grad()
        loss, result = classifier(batch_data.float(), batch_labels)
        loss.backward()
        optimizer.step()   
        
    test_files = table[test_index][:,0]
    test_labels = []
    for file in test_files:
        current_label = 0 if (int(re.search(r'\d{4}', file)[0]) >= 8000) else 1
        test_labels.append(current_label)
        
    test_dataset = SchizophreniaDataset(test_files, 100, test_labels)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=True)  
    
    classifier.eval()
    
    fold_acc_window = []
    fold_acc_file = {}
    for _, batch in enumerate(test_loader):
        batch_data, batch_labels, file_names = batch
        batch_data = batch_data.to(device)
        batch_labels = batch_labels.to(device) 
        
        if(pretrain):
            reps = transformer(batch_data)
            batch_data = reps
        
        loss, result = classifier(batch_data.float(), batch_labels)
        preds = result.argmax(dim=1).detach().cpu().numpy()
        batch_labels = batch_labels.detach().cpu().numpy()
        batch_acc = accuracy_score(batch_labels, preds)
        fold_acc_window.append(batch_acc)
        
        for i in range(batch_labels.shape[0]):
            current_file = file_names[i]
            correct_bool = 1 if preds[i]==batch_labels[i] else 0
            try:
                fold_acc_file[current_file].append(correct_bool)
            except:
                fold_acc_file[current_file] = [correct_bool]
    print(np.mean(fold_acc_window), dict_acc(fold_acc_file))
    overall_w.append(np.mean(fold_acc_window))
    overall_f.append(dict_acc(fold_acc_file))
    
print("Total: ", np.mean(overall_w), np.mean(overall_f))

# setup the transformer model
"""
`options`: a python dictionary containing the following keys:
    ckpt_file: str, a path specifying the pre-trained ckpt file
    load_pretrain: str, ['True', 'False'], whether to load pre-trained weights
    no_grad: str, ['True', 'False'], whether to have gradient flow over this class
    dropout: float/str, use float to modify dropout value during downstream finetune, or use the str `default` for pre-train default values
    spec_aug: str, ['True', 'False'], whether to apply SpecAugment on inputs (used for ASR training)
    spec_aug_prev: str, ['True', 'False'], apply spec augment on input acoustic features if True, else apply on output representations (used for ASR training)
    weighted_sum: str, ['True', 'False'], whether to use a learnable weighted sum to integrate hidden representations from all layers, if False then use the last
    select_layer: int, select from all hidden representations, set to -1 to select the last (will only be used when weighted_sum is False)
    permute_input: str, ['True', 'False'], this attribute is for the forward method. If Ture then input ouput is in the shape of (T, B, D), if False then in (B, T, D)
"""
#options = {
    #'ckpt_file'     : './result/result_transformer/tera/fmllrBase960-F-N-K-libri/states-1000000.ckpt',
    #'load_pretrain' : 'True',
    #'no_grad'       : 'True',
    #'dropout'       : 'default',
    #'spec_aug'      : 'False',
    #'spec_aug_prev' : 'True',
    #'weighted_sum'  : 'False',
    #'select_layer'  : -1,
    #'permute_input' : 'False',
#}
#transformer = TRANSFORMER(options=options, inp_dim=0) # set `inpu_dim=0` to auto load the `inp_dim` from `ckpt_file`

## setup your downstream class model
#classifier = example_classifier(input_dim=768, hidden_dim=128, class_num=2).cuda()

## construct the optimizer
#params = list(transformer.named_parameters()) + list(classifier.named_parameters())
#optimizer = get_optimizer(params=params, lr=4e-3, warmup_proportion=0.7, training_steps=50000)

## forward
#example_inputs = torch.zeros(3, 1200, 40) # A batch of spectrograms:  (batch_size, time_step, feature_size)
## IMPORTANT: Input acoustic features must align with the ones used during our pre-training!
#reps = transformer(example_inputs) # returns: (batch_size, time_step, feature_size)
#labels = torch.LongTensor([0, 1, 0]).cuda()
#loss = classifier(reps, labels)

## update
#loss.backward()
#optimizer.step()

## save
#PATH_TO_SAVE_YOUR_MODEL = 'example.ckpt'
#states = {'Classifier': classifier.state_dict(), 'Transformer': transformer.state_dict()}
## torch.save(states, PATH_TO_SAVE_YOUR_MODEL)