import torch
from transformer.nn_transformer import TRANSFORMER
from downstream.model import example_classifier, RnnClassifier, AvecModel
from downstream.solver import get_optimizer
from downstream.dataloader_ds import AvecDataset, AvecDatasetFull
import pandas as pd
from torch.utils.data import DataLoader
import re
import numpy as np
import torch.nn as nn
import sys
import os
from sklearn.metrics import mean_squared_error
from torch.nn import init
from audtorch.metrics.functional import concordance_cc

device = 'cuda' if torch.cuda.is_available() else 'cpu'

#config = {
                        #'mode'     : 'regression',
                        #'sample_rate' : 1,
                        #'hidden_size'       : 64,
                        #'pre_linear_dims'       : [32], 'post_linear_dims': [32],'drop':0.1,
                        #'concat': 1, 'layers': 3, 'linear': False,
                        #'t_local': 45, 't_global': 150
        #}

#classifier = AvecModel(28, 1, config, 3).to(device)
#features = torch.randn(8,1500, 28)
#labels = torch.randn(8)
#loss, result, correct, valid = classifier.forward(features, labels)

seeds = list(np.random.randint(0,1000,20))
def get_path(participant_ids, processed_path):
    output = []
    for participant_id in participant_ids:
        file_name = str(participant_id) + "_VIDEO.npy"
        output.append(os.path.join(processed_path, file_name))
    return output

def concordance_correlation_coefficient(y_true, y_pred,
                       sample_weight=None,
                       multioutput='uniform_average'):
    cor=np.corrcoef(y_true,y_pred)[0][1]
    
    mean_true=np.mean(y_true)
    mean_pred=np.mean(y_pred)
    
    var_true=np.var(y_true)
    var_pred=np.var(y_pred)
    
    sd_true=np.std(y_true)
    sd_pred=np.std(y_pred)
    
    numerator=2*cor*sd_true*sd_pred
    
    denominator=var_true+var_pred+(mean_true-mean_pred)**2

    return numerator/denominator

output = []
subset = ["gp", "au"]
model_name_flm = "result/result_transformer/flm_base/states-250000.ckpt"
#model_name_au = "result/result_transformer/au_base/states-250000.ckpt"
#model_name_gp = "result/result_transformer/gp_base/states-250000.ckpt"
model_name_au = "result/result_transformer/au_aalbert_3L/states-200000.ckpt"
model_name_gp = "result/result_transformer/gp_base_aalbert/states-200000.ckpt"
model_name_gpau = "result/result_transformer/gpau_aalbert_3L/states-200000.ckpt"
model_name_dict = {"flm":model_name_flm, "au":model_name_au, "gp":model_name_gp}

for seed in seeds:
    torch.manual_seed(seed)
    
    batch_size = 8
    n_steps = 4000
    
    eval_every = 40
    max_len = 5000
    norm_label = False
    
    train_info, dev_info, test_info = "data/train_split.csv", "data/dev_split.csv", "data/test_split.csv"
    regression_col_name = "PHQ_Score"
    regression_col = list(pd.read_csv(test_info).columns).index(regression_col_name)
    processed_npy_path = "/shares/perception-working/minh/avec_processed_three_fps/"
    
    train_paths = get_path(pd.read_csv(train_info).values[:,0], processed_npy_path)
    train_scores = pd.read_csv(train_info).values[:,regression_col]
    dev_paths = get_path(pd.read_csv(dev_info).values[:,0], processed_npy_path)
    dev_scores = pd.read_csv(dev_info).values[:,regression_col]
    test_paths = get_path(pd.read_csv(test_info).values[:,0], processed_npy_path)
    test_scores = pd.read_csv(test_info).values[:,regression_col]
    
    if(norm_label):    
        train_scores = (np.array(train_scores) - 13.5)/27
        dev_scores = (np.array(dev_scores) - 13.5)/27
        test_scores = (np.array(test_scores) - 13.5)/27
    
    
    train_dataset = AvecDatasetFull(train_paths, list(train_scores), max_len=max_len)
    dev_dataset = AvecDatasetFull(dev_paths, dev_scores, max_len=max_len)
    test_dataset = AvecDatasetFull(test_paths, test_scores, max_len=max_len)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    
    test_id_score = {}
    participant_test_id = pd.read_csv(test_info).values[:,0]
    participant_test_scores = pd.read_csv(test_info).values[:,regression_col]
    for i  in range(len(participant_test_id)):
        test_id_score[participant_test_id[i]] = participant_test_scores[i]
    
    dev_id_score = {}
    participant_dev_id = pd.read_csv(dev_info).values[:,0]
    participant_dev_scores = pd.read_csv(dev_info).values[:,regression_col]
    for i  in range(len(participant_dev_id)):
        dev_id_score[participant_dev_id[i]] = participant_dev_scores[i]
    
    
    epochs = n_steps//len(train_loader)
    pretrain_option = True
    
    if(pretrain_option):
        for i in range(1):
            dev_test_scores = {}
            dev_score_break_down = {}
            if(pretrain_option):
                dim_dict = {"flm":272, "gp":88, "au":136}
                inp_dim = sum([dim_dict[x] for x in subset])
            else:
                dim_dict = {"flm":136, "gp":11, "au":17}
                inp_dim = sum([dim_dict[x] for x in subset])
                
            config = {
                        'mode'     : 'regression',
                        'sample_rate' : 1,
                        'hidden_size'       : 64,
                        'pre_linear_dims'       : [32], 'post_linear_dims': [32],'drop':0.1,
                        'concat': 1, 'layers': 3, 'linear': False,
                        't_local': 45, 't_global': 150
                    }        
            
            
            options = {
                
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
                options['ckpt_file'] = model_name_dict[modal]
                current_transformer = TRANSFORMER(options=options, inp_dim=0).to(device)
                current_transformer.train()
                models_dict[modal] = current_transformer            
            
            
            # setup your downstream class model
            classifier = AvecModel(inp_dim, 1, config, seed).to(device)
            classifier.train()
            param_list = []
            for modal in subset:
                param_list += list(models_dict[modal].parameters())
            param_list += list(classifier.parameters())
            optimizer = torch.optim.AdamW(param_list, lr=3e-4)
            
            train_losses = []
            for e in range(epochs):
                num_step_per_epochs = len(train_loader)
                
                for k, batch in enumerate(train_loader):
                    batch_data, batch_scores, participant_id = batch
                    #batch_data = batch_data.to(device)
                    batch_scores = batch_scores.to(device)
                    
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
                    loss, result, correct, valid = classifier.forward(batch_data.float(), batch_scores.float(), valid_lengths)
                    if(loss.item() != loss.item()):
                        continue                    
                    train_losses.append(loss.item())
                    loss.backward()
                    optimizer.step()    
                    
                    current_step = e * num_step_per_epochs + k
                    if(current_step % eval_every == 0):      
                        classifier.eval()
                        if(pretrain_option):
                            for modal in subset:
                                models_dict[modal].eval()
                                
                        fold_preds = []
                        fold_true = []
                        file_id_scores_dev = {}
                        with torch.no_grad():
                            for _, batch in enumerate(dev_loader):
                                batch_data, batch_scores, file_names = batch
                                #batch_data = batch_data.to(device)
                                batch_scores = batch_scores.to(device)
                    
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
                    
                    
                                loss, result, correct, valid = classifier.forward(batch_data.float(), batch_scores.float(), valid_lengths)
                                fold_preds.append(result)
                                fold_true.append(batch_scores.detach().cpu())
                                result_true_dev_np = np.array(batch_scores.detach().cpu())
                                result_pred_dev_np = np.array(result)
                                for l in range(len(result_true_dev_np)):
                                    try:
                                        file_id_scores_dev[file_names[l].item()].append(result_pred_dev_np[l])
                                    except:
                                        file_id_scores_dev[file_names[l].item()] = [result_pred_dev_np[l]]      
                                
                                                        
                                
                        pred_combine = torch.cat(fold_preds, dim=0).reshape(-1,1)
                        true_combine = torch.cat(fold_true, dim=0).reshape(-1,1)
                        val_mse_loss = torch.sum((pred_combine-true_combine)**2)
                        
                        pred_by_id_val = []
                        true_by_id_val = []
                        
                        for dev_id in file_id_scores_dev.keys():
                            true_score = dev_id_score[dev_id]
                            pred_score = np.mean(file_id_scores_dev[dev_id])
                            pred_by_id_val.append(pred_score)
                            true_by_id_val.append(true_score)                    
                        dev_rmse = mean_squared_error(true_by_id_val, pred_by_id_val, squared=False)
                        dev_ccc = concordance_correlation_coefficient(true_by_id_val, np.array(pred_by_id_val))                    
                        dev_score = -abs(dev_ccc) + dev_rmse/100
                        dev_score_break_down[dev_score] = [dev_ccc, dev_rmse]
                        
                        fold_preds_test = []
                        fold_true_test = []
                        file_id_scores = {}
                        
                        with torch.no_grad():
                            for _, batch in enumerate(test_loader):
                                batch_data, batch_scores, file_names = batch
                                #batch_data = batch_data.to(device)
                                batch_scores = batch_scores.to(device)
                    
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
                    
                    
                                loss, result, correct, valid = classifier.forward(batch_data.float(), batch_scores.float(), valid_lengths)
                                fold_preds_test.append(result)
                                fold_true_test.append(batch_scores.detach().cpu())
                                result_true_np = np.array(batch_scores.detach().cpu())
                                result_pred_np = np.array(result)
                                for l in range(len(result_true_np)):
                                    try:
                                        file_id_scores[file_names[l].item()].append(result_pred_np[l])
                                    except:
                                        file_id_scores[file_names[l].item()] = [result_pred_np[l]]
                                
                    
                        pred_by_id = []
                        true_by_id = []
                        for test_id in file_id_scores.keys():
                            true_score = test_id_score[test_id]
                            pred_score = np.mean(file_id_scores[test_id])
                            pred_by_id.append(pred_score)
                            true_by_id.append(true_score)
                            
                        pred_by_id = np.array(pred_by_id, dtype=float)
                        true_by_id = np.array(true_by_id, dtype=float)
                        try:
                            
                            test_rmse = mean_squared_error(true_by_id, pred_by_id, squared=False)
                            test_ccc = concordance_cc(torch.from_numpy(true_by_id), torch.from_numpy(np.array(pred_by_id)))
                            print("Step ", current_step, "Train CCC: ", np.round(np.mean(train_losses),2), "Dev Loss: ", np.round(dev_rmse,2), np.round(dev_ccc,2), \
                                  "Test RMSE: ", np.round(test_rmse,2), "Test CCC: ", np.round(test_ccc.item(),2))
                            dev_test_scores[dev_score] = [test_rmse, test_ccc]
                            train_losses = []
                            classifier.train()
                            if(pretrain_option):
                                for modal in subset:
                                    models_dict[modal].train()                              
                        except:
                            train_losses = []
                            pass
            chosen_stats = dev_test_scores[min(dev_test_scores)]
            chosen_dev_scores = dev_score_break_down[min(dev_test_scores)]
            print("BEST PERFORMING SCORES: ", model_name, chosen_stats)
            output.append([seed, model_name, chosen_dev_scores[0], chosen_dev_scores[1], chosen_stats[0], chosen_stats[1].item()])

    pretrain_option = False
    if(not pretrain_option):
        dev_test_scores = {}
        dev_score_break_down = {}
        config = {
                    'mode'     : 'regression',
                    'sample_rate' : 1,
                    'hidden_size'       : 64,
                    'pre_linear_dims'       : [32], 'post_linear_dims': [32],'drop':0.1,
                    'concat': 1, 'layers': 3, 'linear': False,
                    't_local': 45, 't_global': 150
                }        
        if(pretrain_option):
            dim_dict = {"flm":272, "gp":88, "au":136}
            inp_dim = sum([dim_dict[x] for x in subset])
        else:
            dim_dict = {"flm":136, "gp":11, "au":17}
            inp_dim = sum([dim_dict[x] for x in subset])
            
        # setup your downstream class model
        classifier = AvecModel(inp_dim, 1, config, seed).to(device)
        classifier.train()
        # construct the optimizer
        params = list(list(classifier.named_parameters()))
        #optimizer = get_optimizer(params=params, lr=4e-3, warmup_proportion=0.7, training_steps=25000)     
        optimizer = torch.optim.AdamW(list(classifier.parameters()), lr=3e-4)
        
        for e in range(epochs):
            num_step_per_epochs = len(train_loader)
            for k, batch in enumerate(train_loader):
                batch_data, batch_scores, participant_id = batch
                #batch_data = batch_data.to(device)
                batch_scores = batch_scores.to(device)
    
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
                loss, result, correct, valid = classifier.forward(batch_data.float(), batch_scores.float(), valid_lengths)
                if(loss.item() == loss.item()):
                    loss.backward()
                    optimizer.step()    
    
                current_step = e * num_step_per_epochs + k
                if(current_step % eval_every == 0):      
                    classifier.eval()
                    fold_preds = []
                    fold_true = []
                    file_id_scores_dev = {}
                    with torch.no_grad():
                        for _, batch in enumerate(dev_loader):
                            batch_data, batch_scores, file_names = batch
                            #batch_data = batch_data.to(device)
                            batch_scores = batch_scores.to(device)
    
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
    
    
                            loss, result, correct, valid = classifier.forward(batch_data.float(), batch_scores.float(), valid_lengths)
                            fold_preds.append(result)
                            fold_true.append(batch_scores.detach().cpu())
                            result_true_dev_np = np.array(batch_scores.detach().cpu())
                            result_pred_dev_np = np.array(result)
                            for l in range(len(result_true_dev_np)):
                                try:
                                    file_id_scores_dev[file_names[l].item()].append(result_pred_dev_np[l])
                                except:
                                    file_id_scores_dev[file_names[l].item()] = [result_pred_dev_np[l]]      
    
    
    
                    pred_combine = torch.cat(fold_preds, dim=0).reshape(-1,1)
                    true_combine = torch.cat(fold_true, dim=0).reshape(-1,1)
                    val_mse_loss = torch.sum((pred_combine-true_combine)**2)
    
                    pred_by_id_val = []
                    true_by_id_val = []
                    for dev_id in file_id_scores_dev.keys():
                        true_score = dev_id_score[dev_id]
                        pred_score = np.mean(file_id_scores_dev[dev_id])
                        pred_by_id_val.append(pred_score)
                        true_by_id_val.append(true_score)                    
                    dev_rmse = mean_squared_error(true_by_id_val, pred_by_id_val, squared=False)
                    dev_ccc = concordance_correlation_coefficient(true_by_id_val, np.array(pred_by_id_val))                    
                    dev_score = -abs(dev_ccc) + dev_rmse/100
                    dev_score_break_down[dev_score] = [dev_ccc, dev_rmse]
                    
                    fold_preds_test = []
                    fold_true_test = []
                    file_id_scores = {}
    
                    with torch.no_grad():
                        for _, batch in enumerate(test_loader):
                            batch_data, batch_scores, file_names = batch
                            #batch_data = batch_data.to(device)
                            batch_scores = batch_scores.to(device)
    
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
    
    
                            loss, result, correct, valid = classifier.forward(batch_data.float(), batch_scores.float(), valid_lengths)
                            fold_preds_test.append(result)
                            fold_true_test.append(batch_scores.detach().cpu())
                            result_true_np = np.array(batch_scores.detach().cpu())
                            result_pred_np = np.array(result)
                            for l in range(len(result_true_np)):
                                try:
                                    file_id_scores[file_names[l].item()].append(result_pred_np[l])
                                except:
                                    file_id_scores[file_names[l].item()] = [result_pred_np[l]]
    
    
                    pred_by_id = []
                    true_by_id = []
                    for test_id in file_id_scores.keys():
                        true_score = test_id_score[test_id]
                        pred_score = np.mean(file_id_scores[test_id])
                        pred_by_id.append(pred_score)
                        true_by_id.append(true_score)
    
                    pred_by_id = np.array(pred_by_id, dtype=float)
                    true_by_id = np.array(true_by_id, dtype=float)
                    try:
    
                        test_rmse = mean_squared_error(true_by_id, pred_by_id, squared=False)
                        test_ccc = concordance_cc(torch.from_numpy(true_by_id), torch.from_numpy(np.array(pred_by_id)))
                        print("Step ", current_step, "Dev MSE: ", dev_score, \
                                  "Test RMSE: ", test_rmse, "Test CCC: ", test_ccc.item())
                        dev_test_scores[dev_score] = [test_rmse, test_ccc]
                        classifier.train()
                    except:
                        classifier.train()
                        pass    
         
        chosen_stats = dev_test_scores[min(dev_test_scores)]
        chosen_dev_scores = dev_score_break_down[min(dev_test_scores)]
        print("BEST PERFORMING SCORES: ", chosen_stats)
        output.append([seed, "N/A", chosen_dev_scores[0], chosen_dev_scores[1], chosen_stats[0], chosen_stats[1].item()])
pd.DataFrame(output).to_csv("avec_seed_results_flm.csv", header=None, index=False)
            



