import torch
from transformer.nn_transformer import TRANSFORMER
from downstream.model import RnnClassifierMosi
from downstream.solver import get_optimizer
from downstream.dataloader_ds import MosiDataset
import pandas as pd
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader
import re
import numpy as np
import torch.nn as nn
import sys
from sklearn.metrics import mean_absolute_error
sys.path.append("/home/mtran/CMU-MultimodalSDK")
from mmsdk import mmdatasdk

device = 'cuda' if torch.cuda.is_available() else 'cpu'
subset = ["au"]

mosi_path = "/home/mtran/cmumosi"
dataset = mmdatasdk.mmdataset(mosi_path)
user_ids = np.array(list(dataset['Opinion Segment Labels'].keys()))
score_file = "data/mosi_segment_scores.csv"

path2npy = "/home/mtran/mosi_3fps_npy/"

model_name_flm = "../GoogleDrive/flm_models/states-250000.ckpt"
model_name_au = "../GoogleDrive/flm_models/au_base.ckpt"
model_name_gp = "../GoogleDrive/flm_models/gp_base.ckpt"
model_name_dict = {"flm":model_name_flm, "au":model_name_au, "gp":model_name_gp}

pretrain_option = True

if(pretrain_option):
    dim_dict = {"flm":272, "gp":88, "au":136}
    inp_dim = sum([dim_dict[x] for x in subset])
else:
    dim_dict = {"flm":136, "gp":11, "au":17}
    inp_dim = sum([dim_dict[x] for x in subset])
    
seed = 6
torch.manual_seed(seed)
n_val = 15

kf = KFold(n_splits=5, shuffle=True, random_state=seed)
bs = 8

epochs = 10
eval_every = 40

    
config = {
            'mode'     : 'regression',
            'sample_rate' : 1,
            'hidden_size'       : 64,
            'pre_linear_dims'       : [32], 'post_linear_dims': [32],'drop':0.1,
            'concat': 1, 'layers': 3, 'linear': False,
        }
overall_performance = []
for train_index, test_index in kf.split(user_ids):
    train_ids = user_ids[train_index[:-n_val]]
    test_ids = user_ids[test_index]
    val_ids = user_ids[train_index[-n_val:]]
    
    data_train = MosiDataset(train_ids, score_file, path2npy, subset)
    data_test = MosiDataset(test_ids, score_file, path2npy, subset)
    data_dev = MosiDataset(val_ids, score_file, path2npy, subset)
    
    train_loader = DataLoader(data_train, batch_size=bs, shuffle=True)
    test_loader = DataLoader(data_test, batch_size=bs, shuffle=False)
    dev_loader = DataLoader(data_dev, batch_size=bs, shuffle=False)
    
    #load transformer if pretrained
    if(pretrain_option):
        #'ckpt_file'     : model_name_flm,
        options_flm = {
            
            'load_pretrain' : 'True',
            'no_grad'       : 'True',
            'dropout'       : 'default',
            'spec_aug'      : 'False',
            'spec_aug_prev' : 'True',
            'weighted_sum'  : 'False',
            'select_layer'  : -1,
            'permute_input' : 'False',
        }
        models_dict = {}
        for modal in subset:
            options_flm['ckpt_file'] = model_name_dict[modal]
            current_transformer = TRANSFORMER(options=options_flm, inp_dim=0).to(device)
            current_transformer.train()
            models_dict[modal] = current_transformer
        
        # setup your downstream class model
        classifier = RnnClassifierMosi(inp_dim, 1, config, seed).to(device)
        classifier.train()
        
        param_list = []
        for modal in subset:
            param_list += list(models_dict[modal].parameters())
        param_list += list(classifier.parameters())
        optimizer = torch.optim.AdamW(param_list, lr=3e-4)
        
    else:
        classifier = RnnClassifierMosi(inp_dim, 1, config, seed).to(device)
        optimizer = torch.optim.AdamW(list(classifier.parameters()), lr=3e-4)    
        classifier.train()
    
    dev_scores_dict = {}
    for e in range(epochs):
        num_step_per_epochs = len(train_loader)
        for k, batch in enumerate(train_loader):
            batch_data, batch_scores, participant_id = batch
            #batch_data = batch_data.to(device)
            batch_scores = batch_scores.to(device)
            #if(len(batch_data) == 0):
                #continue
            
            if(pretrain_option):
                reps_dict = {}
                for modal in subset:
                    current_rep = models_dict[modal](batch_data[modal].to(device))
                    reps_dict[modal] = current_rep
                if(len(subset) == 1):
                    batch_data = reps_dict[subset[0]].to(device)
                else:
                    batch_data = torch.cat([reps_dict[x] for x in subset], dim=-1).to(device)
            else:
                if(len(subset) == 1):
                    batch_data = batch_data[subset[0]].to(device)
                else:
                    batch_data = torch.cat([batch_data[x] for x in subset], dim=-1).to(device) 
            
            label_mask = (batch_data.sum(dim=-1) != 0).type(torch.LongTensor).to(device=device, dtype=torch.long)
            valid_lengths = label_mask.sum(dim=1)        
            
            optimizer.zero_grad()
            try:
                loss, result, correct, valid = classifier.forward(batch_data.float(), batch_scores.float(), valid_lengths)
            except:
                continue
            if(loss.item() != loss.item()):
                continue            
            #train_losses.append(loss.item())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(classifier.parameters(), 5)
            optimizer.step()  
            
            current_step = e * num_step_per_epochs + k
            if(current_step % eval_every == 0):   
                classifier.eval()
                if(pretrain_option):
                    for modal in subset:
                        models_dict[modal].eval()
                    
                preds = []
                ground_truth = []
                with torch.no_grad():
                    for _, batch in enumerate(dev_loader):                
                        batch_data, batch_scores, file_names = batch
                        #batch_data = batch_data.to(device)
                        batch_scores = batch_scores.to(device)
                        #if(len(batch_data) == 0):
                            #continue                        
            
                        if(pretrain_option):
                            reps_dict = {}
                            for modal in subset:
                                current_rep = models_dict[modal](batch_data[modal].to(device))
                                reps_dict[modal] = current_rep
                            if(len(subset) == 1):
                                batch_data = reps_dict[subset[0]].to(device)
                            else:
                                batch_data = torch.cat([reps_dict[x] for x in subset], dim=-1).to(device)
                        else:
                            if(len(subset) == 1):
                                batch_data = batch_data[subset[0]].to(device)
                            else:
                                batch_data = torch.cat([batch_data[x] for x in subset], dim=-1).to(device) 
                        
                        label_mask = (batch_data.sum(dim=-1) != 0).type(torch.LongTensor).to(device=device, dtype=torch.long)
                        valid_lengths = label_mask.sum(dim=1)                             
                        
                        loss, result, correct, valid = classifier.forward(\
                            batch_data.float(), batch_scores.float(), valid_lengths)
                        current_preds = list(result.detach().cpu().numpy())
                        current_groundtruth = list(batch_scores.detach().cpu().numpy())
                        if(len(current_preds) == len(current_groundtruth)):
                            preds += current_preds
                            ground_truth += current_groundtruth
                        
                dev_mae = mean_absolute_error(ground_truth, preds)
                dev_cor = np.corrcoef(preds, ground_truth)
                    
                    
                preds_test = []
                ground_truth_test = []
                with torch.no_grad():
                    for _, batch in enumerate(test_loader):                
                        batch_data, batch_scores, file_names = batch
                        #batch_data = batch_data.to(device)
                        batch_scores = batch_scores.to(device)
                        #if(len(batch_data) == 0):
                            #continue
                        
                        if(pretrain_option):
                            reps_dict = {}
                            for modal in subset:
                                current_rep = models_dict[modal](batch_data[modal].to(device))
                                reps_dict[modal] = current_rep
                            if(len(subset) == 1):
                                batch_data = reps_dict[subset[0]].to(device)
                            else:
                                batch_data = torch.cat([reps_dict[x] for x in subset], dim=-1).to(device)
                        else:
                            if(len(subset) == 1):
                                batch_data = batch_data[subset[0]].to(device)
                            else:
                                batch_data = torch.cat([batch_data[x] for x in subset], dim=-1).to(device) 
                            
                        label_mask = (batch_data.sum(dim=-1) != 0).type(torch.LongTensor).to(device=device, dtype=torch.long)
                        valid_lengths = label_mask.sum(dim=1)                         
                        
                        loss, result, correct, valid = classifier.forward(\
                                        batch_data.float(), batch_scores.float(), valid_lengths)
                        current_preds = list(result.detach().cpu().numpy())
                        current_groundtruth = list(batch_scores.detach().cpu().numpy())
                        if(len(current_preds) == len(current_groundtruth)):
                            preds_test += current_preds
                            ground_truth_test += current_groundtruth
                test_mae = mean_absolute_error(ground_truth_test, preds_test)
                test_cor = np.corrcoef(preds_test, ground_truth_test)
                classifier.train()
                if(pretrain_option):
                    for modal in subset:
                        models_dict[modal].train()
                        
                print("Step: ", current_step, " Dev scores: ", dev_mae, abs(dev_cor[0][1]), \
                      " Test scores: ", test_mae, abs(test_cor[0][1]))
                dev_scores_dict[dev_mae/10 - abs(dev_cor[0][1])] = [test_mae, abs(test_cor[0][1])]
                
    print("Fold Performance: ", dev_scores_dict[min(dev_scores_dict)])
    overall_performance.append(dev_scores_dict[min(dev_scores_dict)])
print("Overall ACC: ", np.mean(np.array(overall_performance), axis=0))

#seed6 baseline Overall ACC:  [1.4761626  0.13200975]
#seed 6 pretrain Overall ACC:  [1.44185184 0.17642735]