import torch
from transformer.nn_transformer import TRANSFORMER
from downstream.model import example_classifier, RnnClassifier, FeedForwardClassifier
from downstream.solver import get_optimizer
from downstream.dataloader_ds import SchizophreniaDataset, SchizophreniaSegmentDataset
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold
from torch.utils.data import DataLoader
import re
from sklearn.metrics import accuracy_score
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


subsets = ["watch", "describe", "feel"]
model_name_dict = {
    "result/result_transformer/flm_d256_wdev/model_d256_dev.ckpt":256,
    "result/result_transformer/flm_d512_m25_c12/states-500000.ckpt":512,
    "result/result_transformer/flm_full_d272_wdev/model_d272_dev.ckpt":272,
    "result/result_transformer/flm_full_d272_wdev_25mask/states-500000.ckpt":272,
    } 
seeds = list(range(20))
drugconds = ["PL","OT"]
pretrain_option = [1]
#subset = sys.argv[1]
#model_name = sys.argv[2]
#seed = int(sys.argv[3])
#inp_dim = int(sys.argv[4])
#drugcond = sys.argv[5]

output = []
for seed in seeds:
    for subset in subsets:
        for drugcond in drugconds:
            for pretrain_num in pretrain_option:
                pretrain = True
                if(pretrain):
                    for model_name in model_name_dict.keys():
                        inp_dim = model_name_dict[model_name]
                    
                        config = {
                                    'mode'     : 'classification',
                                    'sample_rate' : 1,
                                    'hidden_size'       : 128,
                                    'pre_linear_dims'       : [20], 'post_linear_dims': [20],'drop':0.2,
                                    'concat': 1, 'layers': 3, 'linear': False,
                                }
                        
                        
                        
                        torch.manual_seed(seed)
                        
                        n_fold = 5
                        sets = ["data/train-clean-schz_chunk_0.csv","data/train-clean-schz_chunk_1.csv",\
                                "data/train-clean-schz_chunk_2.csv","data/train-clean-schz_chunk_3.csv"]
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
                        segment_size = 800
                        bs = 12
                        val_every = 40
                        
                        pretrain = False
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
                                
                            
                            train_dataset = SchizophreniaSegmentDataset(train_files, train_labels, max_len=segment_size)
                            train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True)
                            
                            dev_dataset = SchizophreniaSegmentDataset(val_files, dev_labels, max_len=segment_size)
                            dev_loader = DataLoader(dev_dataset, batch_size=bs, shuffle=True)  
                            
                            fold_dev_test_acc = {}
                            ###########################
                            #load transformer if pretrained
                            if(pretrain):
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
                                classifier = RnnClassifier(inp_dim, 2, config).to(device)
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
                                
                            for e in range(epochs):
                                num_step_per_epochs = len(train_loader)
                                for k, batch in enumerate(train_loader):
                                    batch_data, batch_labels, file_names = batch
                                    batch_data = batch_data.to(device)
                                    batch_labels = duplicate(batch_labels, segment_size, d_bool=False).to(device)
                                    
                                    if(pretrain):
                                        reps = transformer(batch_data)
                                        batch_data = reps
                                    
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
                                                    
                                                label_mask = (batch_data.sum(dim=-1) != 0).type(torch.LongTensor).to(device=device, dtype=torch.long)
                                                valid_lengths = label_mask.sum(dim=1)  
                                                loss, result, correct, valid = classifier.forward(batch_data.float(), batch_labels, valid_lengths)
                                                batch_acc = correct.item()/valid.item() if val_acc > 0.35 else 1-correct.item()/valid.item()
                                                fold_acc_window_test.append(batch_acc)
                                                
                                        classifier.train()
                                        if(pretrain):
                                            transformer.train()
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
                    inp_dim = 100
                    
                    config = {
                                'mode'     : 'classification',
                                'sample_rate' : 1,
                                'hidden_size'       : 128,
                                'pre_linear_dims'       : [20], 'post_linear_dims': [20],'drop':0.2,
                                'concat': 1, 'layers': 3, 'linear': False,
                            }
                    
                    
                    
                    torch.manual_seed(seed)
                    
                    n_fold = 5
                    sets = ["data/train-clean-schz_chunk_0.csv","data/train-clean-schz_chunk_1.csv",\
                            "data/train-clean-schz_chunk_2.csv","data/train-clean-schz_chunk_3.csv"]
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
                    segment_size = 800
                    bs = 12
                    val_every = 40
                    
                    pretrain = False
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
                            
                        
                        train_dataset = SchizophreniaSegmentDataset(train_files, train_labels, max_len=segment_size)
                        train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True)
                        
                        dev_dataset = SchizophreniaSegmentDataset(val_files, dev_labels, max_len=segment_size)
                        dev_loader = DataLoader(dev_dataset, batch_size=bs, shuffle=True)  
                        
                        fold_dev_test_acc = {}
                        ###########################
                        #load transformer if pretrained
                        if(pretrain):
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
                            classifier = RnnClassifier(inp_dim, 2, config).to(device)
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
                            
                        for e in range(epochs):
                            num_step_per_epochs = len(train_loader)
                            for k, batch in enumerate(train_loader):
                                batch_data, batch_labels, file_names = batch
                                batch_data = batch_data.to(device)
                                batch_labels = duplicate(batch_labels, segment_size, d_bool=False).to(device)
                                
                                if(pretrain):
                                    reps = transformer(batch_data)
                                    batch_data = reps
                                
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
                                                
                                            label_mask = (batch_data.sum(dim=-1) != 0).type(torch.LongTensor).to(device=device, dtype=torch.long)
                                            valid_lengths = label_mask.sum(dim=1)  
                                            loss, result, correct, valid = classifier.forward(batch_data.float(), batch_labels, valid_lengths)
                                            batch_acc = correct.item()/valid.item() if val_acc > 0.35 else 1-correct.item()/valid.item()
                                            fold_acc_window_test.append(batch_acc)
                                            
                                    classifier.train()
                                    if(pretrain):
                                        transformer.train()
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
                    
pd.DataFrame(output).to_csv("multiple_seed_schz_clf_results.csv", header=None, index=False)
                            

#123 -> 0.7283333333333333 / 0.51380952381
