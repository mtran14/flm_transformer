import torch
from transformer.nn_transformer import TRANSFORMER
from downstream.model import example_classifier, RnnClassifier, FeedForwardClassifier
from downstream.solver import get_optimizer
from downstream.dataloader_ds import SchizophreniaMMDataset
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold
from torch.utils.data import DataLoader
import re
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
from sklearn.linear_model import LogisticRegression
import torch.nn as nn
import sys
device = 'cuda' if torch.cuda.is_available() else 'cpu'

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

def filter_files(root_files, list_files, drugCond=None):
    valid_info = []
    if(drugCond is not None):
        drugcond_data = pd.read_excel("data/PORQ_drugUnblinding.xlsx").values
        for row in drugcond_data:
            if(row[3] == drugCond):
                valid_info.append(row[:-1])
                
    output_files = []
    for file in root_files:
        src_info = file.split("/")[-1].split(".npy")[0]
        for pos_file in list_files:
            if(src_info not in pos_file):
                continue
            participant_id = int(re.search(r'\d{4}', file)[0])
            if(participant_id >= 8000):
                output_files.append(pos_file)
                continue
            
            keep = False
            for drugcondinf in valid_info:
                if(str(drugcondinf[0]) in pos_file and \
                   "Day" + str(drugcondinf[1]) in pos_file and \
                   str(drugcondinf[2])[:10] in pos_file):
                    keep = True
            if(keep):
                output_files.append(pos_file)
    return output_files

def reverse_pred(input_list):
    output_list = []
    for x in range(len(input_list)):
        if(x == 0):
            output_list.append(1)
        else:
            output_list.append(0)


#subsets = ["watch", "describe", "feel"]
subsets = ["watch", "describe", "feel"]
#model_name_dict = {
    #"result/result_transformer/flm_small/states-250000.ckpt":272,
    #"result/result_transformer/flm_base/states-250000.ckpt":272,
    #"result/result_transformer/flm_large_1mask/best_160_save.ckpt":544,
    #"result/result_transformer/flm_large/states-250000.ckpt":544,
    #} 
    
#model_name_flm = "../GoogleDrive/flm_models/states-250000.ckpt"
#model_name_au = "../GoogleDrive/flm_models/au_base.ckpt"
#model_name_gp = "../GoogleDrive/flm_models/gp_base.ckpt"
model_name_flm = "result/result_transformer/flm_base/states-250000.ckpt"
#model_name_au = "result/result_transformer/au_base/states-250000.ckpt"
#model_name_gp = "result/result_transformer/gp_base/states-250000.ckpt"
model_name_au = "result/result_transformer/au_aalbert_3L/states-200000.ckpt"
model_name_gp = "result/result_transformer/gp_base_aalbert/states-200000.ckpt"
model_name_gpau = "result/result_transformer/gpau_aalbert_3L/states-200000.ckpt"
model_name_dict = {"flm":model_name_flm, "au":model_name_au, "gp":model_name_gp}

seeds = list(np.random.randint(0,1000,5))
drugconds = ["PL","OT"]
pretrain_option = [False,True]
sources = ["au", "gp"]

output = []
for seed in seeds:
    for subset in subsets:
        for drugcond in drugconds:
            for pretrain in pretrain_option:
                if(pretrain):
                    for i in range(1):
                        if(pretrain):
                            #dim_dict = {"flm":272, "gp":88, "au":136}
                            dim_dict = {"flm":272, "gp":84, "au":120}
                            inp_dim = sum([dim_dict[x] for x in sources])
                        else:
                            dim_dict = {"flm":136, "gp":11, "au":17}
                            inp_dim = sum([dim_dict[x] for x in sources])                        
                    
                        config = {
                                    'mode'     : 'classification',
                                    'sample_rate' : 1,
                                    'hidden_size'       : 128,
                                    'pre_linear_dims'       : [20], 'post_linear_dims': [20],'drop':0.2,
                                    'concat': 1, 'layers': 3, 'linear': False,
                                }
                        
                        
                        
                        torch.manual_seed(seed)
                        
                        n_fold = 5
                        sets = ["data/train-clean-schz_chunk_0.csv","data/train-clean-schz_chunk_1.csv"]
                        tables = [pd.read_csv(s, header=None) for s in sets]
                        table = pd.concat(tables, ignore_index=True).values
                        
                        name_sets = ["data/train-clean-360_chunk_0.csv","data/train-clean-360_chunk_1.csv"]
                        tables_name = [pd.read_csv(s, header=None) for s in name_sets]
                        table_name = pd.concat(tables_name, ignore_index=True).values
                        table_label = []
                        for file in table_name[:,0]:
                            current_label = 0 if (int(re.search(r'\d{4}', file)[0]) >= 8000) else 1
                            table_label.append(current_label)
                        
                        table_filter = []
                        for row in table:
                            if(subset in row[0]):
                                table_filter.append(row)
                        table = np.array(table_filter)
                        
                        kf = StratifiedKFold(n_splits=n_fold, shuffle=True, random_state=seed)
                        n_step = 200
                        n_val = 16
                        segment_size = 100
                        bs = 12
                        val_every = 40
                        
                        epochs = 10
                        
                        overall_w = []
                        overall_f = []
                        for train_index, test_index in kf.split(table_name, table_label):
                            train_files_name = table_name[train_index[:-n_val]][:,0]
                            train_labels = []
                            train_files = filter_files(train_files_name, table[:,0], drugCond=drugcond)
                            
                            val_files_name = table_name[train_index[-n_val:]][:,0]
                            dev_labels = []
                            val_files = filter_files(val_files_name, table[:,0], drugCond=drugcond)
                            
                            for file in train_files:
                                current_label = 0 if (int(re.search(r'\d{4}', file)[0]) >= 8000) else 1
                                train_labels.append(current_label)
                            for file in val_files:
                                current_label = 0 if (int(re.search(r'\d{4}', file)[0]) >= 8000) else 1
                                dev_labels.append(current_label)    
                                
                            
                            train_dataset = SchizophreniaMMDataset(train_files, train_labels, max_len=segment_size)
                            train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True)
                            
                            dev_dataset = SchizophreniaMMDataset(val_files, dev_labels, max_len=segment_size)
                            dev_loader = DataLoader(dev_dataset, batch_size=bs, shuffle=True)  
                            
                            fold_dev_test_acc = {}
                            ###########################
                            #load transformer if pretrained
                            if(pretrain):
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
                                for modal in sources:
                                    options['ckpt_file'] = model_name_dict[modal]
                                    current_transformer = TRANSFORMER(options=options, inp_dim=0).to(device)
                                    current_transformer.train()
                                    models_dict[modal] = current_transformer                                
                                
                                # setup your downstream class model
                                classifier = RnnClassifier(inp_dim, 2, config, seed).to(device)
                                classifier.train()
                                # construct the optimizer
                                param_list = []
                                for modal in sources:
                                    param_list += list(models_dict[modal].parameters())
                                param_list += list(classifier.parameters())
                                optimizer = torch.optim.AdamW(param_list, lr=3e-4)
                                    
                            ###########################
                            else:
                                #init model and optimizer
                                #classifier = example_classifier(input_dim=136, hidden_dim=64, class_num=2).to(device)
                                classifier = RnnClassifier(136, 2, config, seed).to(device)
                                optimizer = torch.optim.AdamW(list(classifier.parameters()), lr=3e-4)    
                                classifier.train()
                            ###########################
                                
                            for e in range(epochs):
                                num_step_per_epochs = len(train_loader)
                                for k, batch in enumerate(train_loader):
                                    batch_data, batch_labels, file_names = batch
                                    #batch_data = batch_data.to(device)
                                    batch_labels = duplicate(batch_labels, segment_size, d_bool=False).to(device)
                                    
                                    if(pretrain):
                                        reps_dict = {}
                                        for modal in sources:
                                            current_rep = models_dict[modal](batch_data[modal].to(device))
                                            reps_dict[modal] = current_rep
                                        if(len(sources) == 1):
                                            batch_data = reps_dict[sources[0]].to(device)
                                        else:
                                            batch_data = torch.cat([reps_dict[x] for x in sources], dim=-1).to(device)
                                    else:
                                        if(len(sources) == 1):
                                            batch_data = batch_data[sources[0]].to(device)
                                        else:
                                            batch_data = torch.cat([batch_data[x] for x in sources], dim=-1).to(device)                                     
                                    
                                    label_mask = (batch_data.sum(dim=-1) != 0).type(torch.LongTensor).to(device=device, dtype=torch.long)
                                    valid_lengths = label_mask.sum(dim=1)        
                                    
                                    optimizer.zero_grad()
                                    loss, result, correct, valid = classifier.forward(batch_data.float(), batch_labels, valid_lengths)
                                    loss.backward()
                                    optimizer.step()  
                                    
                                    current_step = e * num_step_per_epochs + k
                                    if(current_step % val_every == 0):   
                                        classifier.eval()
                                        if(pretrain):
                                            for modal in sources:
                                                models_dict[modal].eval()                                        
                                        fold_acc_window = []
                                        #fold_acc_file = {}
                                        with torch.no_grad():
                                            for _, batch in enumerate(dev_loader):
                                                batch_data, batch_labels, file_names = batch
                                                #batch_data = batch_data.to(device)
                                                batch_labels = duplicate(batch_labels, segment_size, d_bool=False).to(device) 
                                        
                                                if(pretrain):
                                                    reps_dict = {}
                                                    for modal in sources:
                                                        current_rep = models_dict[modal](batch_data[modal].to(device))
                                                        reps_dict[modal] = current_rep
                                                    if(len(sources) == 1):
                                                        batch_data = reps_dict[sources[0]].to(device)
                                                    else:
                                                        batch_data = torch.cat([reps_dict[x] for x in sources], dim=-1).to(device)
                                                else:
                                                    if(len(sources) == 1):
                                                        batch_data = batch_data[sources[0]].to(device)
                                                    else:
                                                        batch_data = torch.cat([batch_data[x] for x in sources], dim=-1).to(device)                                                
                                                    
                                                label_mask = (batch_data.sum(dim=-1) != 0).type(torch.LongTensor).to(device=device, dtype=torch.long)
                                                valid_lengths = label_mask.sum(dim=1)          
                                        
                                                
                                                loss, result, correct, valid = classifier.forward(batch_data.float(), batch_labels, valid_lengths)
                                                preds = result.argmax(dim=1).detach().cpu().numpy()
                                                batch_labels = batch_labels.detach().cpu().numpy()
                                                batch_acc = accuracy_score(batch_labels, preds)
                                                fold_acc_window.append(correct.item()/valid.item())
                                             
                                        val_acc = np.mean(fold_acc_window)
                                        test_files_name = table_name[test_index][:,0]
                                        test_labels = []
                                        test_files = filter_files(test_files_name, table[:,0], drugCond=drugcond)
                                        
                                        for file in test_files:
                                            current_label = 0 if (int(re.search(r'\d{4}', file)[0]) >= 8000) else 1
                                            test_labels.append(current_label)
                                                        
                                        
                                        test_dataset = SchizophreniaMMDataset(test_files, test_labels, max_len=segment_size)
                                        test_loader = DataLoader(test_dataset, batch_size=bs, shuffle=True)  
                                        
                                        
                                        fold_acc_window_test = []
                                        fold_acc_file_test = {}
                                        with torch.no_grad():
                                            for _, batch in enumerate(test_loader):
                                                batch_data, batch_labels, file_names = batch
                                                #batch_data = batch_data.to(device)
                                                batch_labels = duplicate(batch_labels, segment_size, d_bool=False).to(device) 
                                                
                                                if(pretrain):
                                                    reps_dict = {}
                                                    for modal in sources:
                                                        current_rep = models_dict[modal](batch_data[modal].to(device))
                                                        reps_dict[modal] = current_rep
                                                    if(len(sources) == 1):
                                                        batch_data = reps_dict[sources[0]].to(device)
                                                    else:
                                                        batch_data = torch.cat([reps_dict[x] for x in sources], dim=-1).to(device)
                                                else:
                                                    if(len(sources) == 1):
                                                        batch_data = batch_data[sources[0]].to(device)
                                                    else:
                                                        batch_data = torch.cat([batch_data[x] for x in sources], dim=-1).to(device)                                                
                                                    
                                                label_mask = (batch_data.sum(dim=-1) != 0).type(torch.LongTensor).to(device=device, dtype=torch.long)
                                                valid_lengths = label_mask.sum(dim=1)  
                                                loss, result, correct, valid = classifier.forward(batch_data.float(), batch_labels, valid_lengths)
                                                batch_acc = correct.item()/valid.item() if val_acc > 0.35 else 1-correct.item()/valid.item()
                                                fold_acc_window_test.append(batch_acc)
                                                
                                        classifier.train()
                                        if(pretrain):
                                            for modal in sources:
                                                models_dict[modal].train()                                         
                                        print("Dev: ", np.mean(fold_acc_window), "Test: ", np.mean(fold_acc_window_test), \
                                              " P(1):", 1-sum(test_labels)/len(test_labels), " P(0):", sum(test_labels)/len(test_labels))
                                        if(np.mean(fold_acc_window) > 0.35):
                                            fold_dev_test_acc[np.mean(fold_acc_window)] = np.mean(fold_acc_window_test)
                                        else:
                                            fold_dev_test_acc[1-np.mean(fold_acc_window)] = np.mean(fold_acc_window_test)
                            fold_test_acc = fold_dev_test_acc[max(fold_dev_test_acc)] #test acc w/ max dev acc
                            print("Fold Acc: ", fold_test_acc)
                            overall_f.append(fold_test_acc)
                        print(seed, subset, drugcond, pretrain, model_name, "CV Test ACC: ", np.mean(overall_f))
                        output.append([seed, subset, drugcond, pretrain, model_name, np.mean(overall_f)])
                else:
                    model_name = "N/A"
                    if(pretrain):
                        dim_dict = {"flm":272, "gp":88, "au":136}
                        inp_dim = sum([dim_dict[x] for x in sources])
                    else:
                        dim_dict = {"flm":136, "gp":11, "au":17}
                        inp_dim = sum([dim_dict[x] for x in sources])                        
                    
                    config = {
                                'mode'     : 'classification',
                                'sample_rate' : 1,
                                'hidden_size'       : 128,
                                'pre_linear_dims'       : [20], 'post_linear_dims': [20],'drop':0.2,
                                'concat': 1, 'layers': 3, 'linear': False,
                            }
                    
                    
                    
                    torch.manual_seed(seed)
                    
                    n_fold = 5
                    sets = ["data/train-clean-schz_chunk_0.csv","data/train-clean-schz_chunk_1.csv"]
                    tables = [pd.read_csv(s, header=None) for s in sets]
                    table = pd.concat(tables, ignore_index=True).values
                    
                    name_sets = ["data/train-clean-360_chunk_0.csv","data/train-clean-360_chunk_1.csv"]
                    tables_name = [pd.read_csv(s, header=None) for s in name_sets]
                    table_name = pd.concat(tables_name, ignore_index=True).values
                    table_label = []
                    for file in table_name[:,0]:
                        current_label = 0 if (int(re.search(r'\d{4}', file)[0]) >= 8000) else 1
                        table_label.append(current_label)
                    
                    table_filter = []
                    for row in table:
                        if(subset in row[0]):
                            table_filter.append(row)
                    table = np.array(table_filter)                    
                    
                    kf = StratifiedKFold(n_splits=n_fold, shuffle=True, random_state=seed)
                    n_step = 200
                    n_val = 16
                    segment_size = 100
                    bs = 12
                    val_every = 40
                    
                    epochs = 10
                    
                    overall_w = []
                    overall_f = []
                    for train_index, test_index in kf.split(table_name, table_label):
                        train_files_name = table_name[train_index[:-n_val]][:,0]
                        train_labels = []
                        train_files = filter_files(train_files_name, table[:,0], drugCond=drugcond)
                        
                        val_files_name = table_name[train_index[-n_val:]][:,0]
                        dev_labels = []
                        val_files = filter_files(val_files_name, table[:,0], drugCond=drugcond)
                        
                        for file in train_files:
                            current_label = 0 if (int(re.search(r'\d{4}', file)[0]) >= 8000) else 1
                            train_labels.append(current_label)
                        for file in val_files:
                            current_label = 0 if (int(re.search(r'\d{4}', file)[0]) >= 8000) else 1
                            dev_labels.append(current_label)    
                            
                        
                        train_dataset = SchizophreniaMMDataset(train_files, train_labels, max_len=segment_size)
                        train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True)
                        
                        dev_dataset = SchizophreniaMMDataset(val_files, dev_labels, max_len=segment_size)
                        dev_loader = DataLoader(dev_dataset, batch_size=bs, shuffle=True)  
                        
                        fold_dev_test_acc = {}
                        ###########################
                        #load transformer if pretrained
                        if(pretrain):
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
                            for modal in sources:
                                options['ckpt_file'] = model_name_dict[modal]
                                current_transformer = TRANSFORMER(options=options, inp_dim=0).to(device)
                                current_transformer.train()
                                models_dict[modal] = current_transformer                                
                            
                            # setup your downstream class model
                            classifier = RnnClassifier(inp_dim, 2, config, seed).to(device)
                            classifier.train()
                            # construct the optimizer
                            param_list = []
                            for modal in sources:
                                param_list += list(models_dict[modal].parameters())
                            param_list += list(classifier.parameters())
                            optimizer = torch.optim.AdamW(param_list, lr=3e-4)
                                
                        ###########################
                        else:
                            #init model and optimizer
                            #classifier = example_classifier(input_dim=136, hidden_dim=64, class_num=2).to(device)
                            classifier = RnnClassifier(inp_dim, 2, config, seed).to(device)
                            optimizer = torch.optim.AdamW(list(classifier.parameters()), lr=3e-4)    
                            classifier.train()                        
                        ###########################
                            
                        for e in range(epochs):
                            num_step_per_epochs = len(train_loader)
                            for k, batch in enumerate(train_loader):
                                batch_data, batch_labels, file_names = batch
                                #batch_data = batch_data.to(device)
                                batch_labels = duplicate(batch_labels, segment_size, d_bool=False).to(device)
                                
                                if(pretrain):
                                    reps_dict = {}
                                    for modal in sources:
                                        current_rep = models_dict[modal](batch_data[modal].to(device))
                                        reps_dict[modal] = current_rep
                                    if(len(sources) == 1):
                                        batch_data = reps_dict[sources[0]].to(device)
                                    else:
                                        batch_data = torch.cat([reps_dict[x] for x in sources], dim=-1).to(device)
                                else:
                                    if(len(sources) == 1):
                                        batch_data = batch_data[sources[0]].to(device)
                                    else:
                                        batch_data = torch.cat([batch_data[x] for x in sources], dim=-1).to(device)                                                               
                                
                                label_mask = (batch_data.sum(dim=-1) != 0).type(torch.LongTensor).to(device=device, dtype=torch.long)
                                valid_lengths = label_mask.sum(dim=1)        
                                
                                optimizer.zero_grad()
                                loss, result, correct, valid = classifier.forward(batch_data.float(), batch_labels, valid_lengths)
                                loss.backward()
                                optimizer.step()  
                                
                                current_step = e * num_step_per_epochs + k
                                if(current_step % val_every == 0):   
                                    classifier.eval()
                                    if(pretrain):
                                        for modal in sources:
                                            models_dict[modal].eval()                                    
                                    fold_acc_window = []
                                    #fold_acc_file = {}
                                    with torch.no_grad():
                                        for _, batch in enumerate(dev_loader):
                                            batch_data, batch_labels, file_names = batch
                                            #batch_data = batch_data.to(device)
                                            batch_labels = duplicate(batch_labels, segment_size, d_bool=False).to(device) 
                                    
                                            if(pretrain):
                                                reps_dict = {}
                                                for modal in sources:
                                                    current_rep = models_dict[modal](batch_data[modal].to(device))
                                                    reps_dict[modal] = current_rep
                                                if(len(sources) == 1):
                                                    batch_data = reps_dict[sources[0]].to(device)
                                                else:
                                                    batch_data = torch.cat([reps_dict[x] for x in sources], dim=-1).to(device)
                                            else:
                                                if(len(sources) == 1):
                                                    batch_data = batch_data[sources[0]].to(device)
                                                else:
                                                    batch_data = torch.cat([batch_data[x] for x in sources], dim=-1).to(device)                                                                                         
                                                
                                            label_mask = (batch_data.sum(dim=-1) != 0).type(torch.LongTensor).to(device=device, dtype=torch.long)
                                            valid_lengths = label_mask.sum(dim=1)          
                                    
                                            
                                            loss, result, correct, valid = classifier.forward(batch_data.float(), batch_labels, valid_lengths)
                                            preds = result.argmax(dim=1).detach().cpu().numpy()
                                            batch_labels = batch_labels.detach().cpu().numpy()
                                            batch_acc = accuracy_score(batch_labels, preds)
                                            fold_acc_window.append(correct.item()/valid.item())
                                         
                                    val_acc = np.mean(fold_acc_window)
                                    test_files_name = table_name[test_index][:,0]
                                    test_labels = []
                                    test_files = filter_files(test_files_name, table[:,0], drugCond=drugcond)
                                    
                                    for file in test_files:
                                        current_label = 0 if (int(re.search(r'\d{4}', file)[0]) >= 8000) else 1
                                        test_labels.append(current_label)
                                                    
                                    
                                    test_dataset = SchizophreniaMMDataset(test_files, test_labels, max_len=segment_size)
                                    test_loader = DataLoader(test_dataset, batch_size=bs, shuffle=True)  
                                    
                                    
                                    fold_acc_window_test = []
                                    fold_acc_file_test = {}
                                    
                                    pred_all_test = []
                                    label_all_test = []
                                    with torch.no_grad():
                                        for _, batch in enumerate(test_loader):
                                            batch_data, batch_labels, file_names = batch
                                            #batch_data = batch_data.to(device)
                                            batch_labels = duplicate(batch_labels, segment_size, d_bool=False).to(device) 
                                            
                                            if(pretrain):
                                                reps_dict = {}
                                                for modal in sources:
                                                    current_rep = models_dict[modal](batch_data[modal].to(device))
                                                    reps_dict[modal] = current_rep
                                                if(len(sources) == 1):
                                                    batch_data = reps_dict[sources[0]].to(device)
                                                else:
                                                    batch_data = torch.cat([reps_dict[x] for x in sources], dim=-1).to(device)
                                            else:
                                                if(len(sources) == 1):
                                                    batch_data = batch_data[sources[0]].to(device)
                                                else:
                                                    batch_data = torch.cat([batch_data[x] for x in sources], dim=-1).to(device)                                                                                  
                                                
                                            label_mask = (batch_data.sum(dim=-1) != 0).type(torch.LongTensor).to(device=device, dtype=torch.long)
                                            valid_lengths = label_mask.sum(dim=1)  
                                            loss, result, correct, valid = classifier.forward(batch_data.float(), batch_labels, valid_lengths)
                                            batch_acc = correct.item()/valid.item() if val_acc > 0.35 else 1-correct.item()/valid.item()
                                            fold_acc_window_test.append(batch_acc)
                                            
                                            predictions = list(result.argmax(dim=-1).detach().cpu().numpy())
                                            predictions_m = reverse_pred(predictions) if val_acc <= 0.35 else predictions
                                            print(batch_labels)
                                            label_m = list(batch_labels.detach().cpu().numpy())
                                            if(len(predictions_m) == len(label_m)):
                                                pred_all_test += predictions_m
                                                label_all_test += label_m
                                            
                                    classifier.train()
                                    if(pretrain):
                                        for modal in sources:
                                            models_dict[modal].train()                                    
                                    print("Dev: ", np.mean(fold_acc_window), "Test: ", accuracy_score(label_all_test, pred_all_test), f1_score(label_all_test, pred_all_test), \
                                          " P(1):", 1-sum(test_labels)/len(test_labels), " P(0):", sum(test_labels)/len(test_labels))
                                    if(np.mean(fold_acc_window) > 0.35):
                                        fold_dev_test_acc[np.mean(fold_acc_window)] = np.mean(fold_acc_window_test)
                                    else:
                                        fold_dev_test_acc[1-np.mean(fold_acc_window)] = np.mean(fold_acc_window_test)
                        fold_test_acc = fold_dev_test_acc[max(fold_dev_test_acc)] #test acc w/ max dev acc
                        print("Fold Acc: ", fold_test_acc)
                        overall_f.append(fold_test_acc)
                    print(seed, subset, drugcond, pretrain, model_name, "CV Test ACC: ", np.mean(overall_f))
                    output.append([seed, subset, drugcond, pretrain, model_name, np.mean(overall_f)])
                    
pd.DataFrame(output).to_csv("multiple_seed_schz_clf_results_gpau.csv", header=None, index=False)
                            

#123 -> 0.7283333333333333 / 0.51380952381
