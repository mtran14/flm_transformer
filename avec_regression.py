import torch
from transformer.nn_transformer import TRANSFORMER
from downstream.model import example_classifier, RnnClassifier
from downstream.solver import get_optimizer
from downstream.dataloader_ds import AvecDataset
import pandas as pd
from torch.utils.data import DataLoader
import re
import numpy as np
import torch.nn as nn
import sys
import os

device = 'cuda' if torch.cuda.is_available() else 'cpu'
batch_size = 16
pretrain_option = True
model_name_dict = {
    "result/result_transformer/flm_d256_wdev/model_d256_dev.ckpt":256,
    "result/result_transformer/flm_d512_m25_c12/states-500000.ckpt":512,
    "result/result_transformer/flm_full_d272_wdev/model_d272_dev.ckpt":272,
    "result/result_transformer/flm_full_d272_wdev_25mask/states-500000.ckpt":272,
    } 
epochs = 10
eval_every = 40

def get_path(participant_ids, processed_path):
    output = []
    for participant_id in participant_ids:
        file_name = str(participant_id) + "_VIDEO.npy"
        output.append(os.path.join(processed_path, file_name))
    return output


train_info, dev_info, test_info = "data/train_split.csv", "data/dev_split.csv", "data/test_split.csv"
regression_col_name = "PHQ_Score"
regression_col = list(pd.read_csv(test_info).columns).index(regression_col_name)
processed_npy_path = "/shares/perception-working/minh/avec_processed/"

train_paths = get_path(pd.read_csv(train_info).values[:,0], processed_npy_path)
train_scores = pd.read_csv(train_info).values[:,regression_col]
dev_paths = get_path(pd.read_csv(dev_info).values[:,0], processed_npy_path)
dev_scores = pd.read_csv(dev_info).values[:,regression_col]
test_paths = get_path(pd.read_csv(test_info).values[:,0], processed_npy_path)
test_scores = pd.read_csv(test_info).values[:,regression_col]

train_dataset = AvecDataset(train_paths, train_scores)
dev_dataset = AvecDataset(dev_paths, dev_scores)
test_dataset = AvecDataset(test_paths, test_scores)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
dev_loader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

if(pretrain_option):
    for model_name in model_name_dict.keys():
        inp_dim = model_name_dict[model_name]
        config = {
                    'mode'     : 'regression',
                    'sample_rate' : 1,
                    'hidden_size'       : 128,
                    'pre_linear_dims'       : [20], 'post_linear_dims': [20],'drop':0.2,
                    'concat': 1, 'layers': 3, 'linear': False,
                }        
        options = {
            'ckpt_file'     : model_name,
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
        classifier = RnnClassifier(inp_dim, 1, config).to(device)
        # construct the optimizer
        params = list(list(transformer.named_parameters()) + list(classifier.named_parameters()))
        optimizer = get_optimizer(params=params, lr=4e-3, warmup_proportion=0.7, training_steps=30000)        
        
        for e in range(epochs):
            num_step_per_epochs = len(train_loader)
            for k, batch in enumerate(train_loader):
                batch_data, batch_scores, participant_id = batch
                batch_data = batch_data.to(device)
                batch_scores = batch_scores.to(device)
                
                if(pretrain_option):
                    reps = transformer(batch_data)
                    batch_data = reps
                
                label_mask = (batch_data.sum(dim=-1) != 0).type(torch.LongTensor).to(device=device, dtype=torch.long)
                valid_lengths = label_mask.sum(dim=1)        
                
                optimizer.zero_grad()
                loss, result, correct, valid = classifier.forward(batch_data.float(), batch_scores.float(), valid_lengths)
                loss.backward()
                optimizer.step()    
                
                current_step = e * num_step_per_epochs + k
                if(current_step % eval_every == 0):      
                    classifier.eval()
                    if(pretrain):
                        transformer.eval()
                    fold_preds = []
                    fold_true = []
                    
                    with torch.no_grad():
                        for _, batch in enumerate(dev_loader):
                            batch_data, batch_scores, file_names = batch
                            batch_data = batch_data.to(device)
                            batch_scores = batch_scores.to(device)
                
                            if(pretrain_option):
                                reps = transformer(batch_data)
                                batch_data = reps
                
                            label_mask = (batch_data.sum(dim=-1) != 0).type(torch.LongTensor).to(device=device, dtype=torch.long)
                            valid_lengths = label_mask.sum(dim=1)          
                
                
                            loss, result, correct, valid = classifier.forward(batch_data.float(), batch_scores.float(), valid_lengths)
                            fold_preds.append(result)
                            fold_true.append(batch_scores.detach().cpu())
                            
                    pred_combine = torch.cat(fold_preds, dim=0).reshape(-1,1)
                    true_combine = torch.cat(fold_true, dim=0).reshape(-1,1)
                    val_mse_loss = torch.sum((pred_combine-true_combine)**2)
                    
                    fold_preds_test = []
                    fold_true_test = []
                    
                    with torch.no_grad():
                        for _, batch in enumerate(test_loader):
                            batch_data, batch_scores, file_names = batch
                            batch_data = batch_data.to(device)
                            batch_scores = batch_scores.to(device)
                
                            if(pretrain_option):
                                reps = transformer(batch_data)
                                batch_data = reps
                
                            label_mask = (batch_data.sum(dim=-1) != 0).type(torch.LongTensor).to(device=device, dtype=torch.long)
                            valid_lengths = label_mask.sum(dim=1)          
                
                
                            loss, result, correct, valid = classifier.forward(batch_data.float(), batch_scores.float(), valid_lengths)
                            fold_preds_test.append(result)
                            fold_true_test.append(batch_scores.detach().cpu())
                
                    pred_combine_test = torch.cat(fold_preds_test, dim=0).reshape(-1,1)
                    true_combine_test = torch.cat(fold_true_test, dim=0).reshape(-1,1)
                    test_mse_loss = torch.sum((pred_combine_test-true_combine_test)**2)    
                    print("Step ", current_step, "Dev MSE: ", val_mse_loss, "Test MSE: ", test_mse_loss)
                    classifier.train()
                    if(pretrain_option):
                        transformer.train()                    
                    
        
        
        
print("here")



