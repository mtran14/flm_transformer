import torch
from transformer.nn_transformer import TRANSFORMER, SCHZ_TRANSFORMER
from transformer.model import TransformerForMaskedAcousticModel
from downstream.solver import get_optimizer
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold
from torch.utils.data import DataLoader
import re
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
import numpy as np
from sklearn.linear_model import LogisticRegression
import torch.nn as nn
import sys
import os
from transformer.mam import process_train_MAM_data
from run_upstream import get_upstream_args
from transformer.runner import Runner
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def process_data(spec):
    """Process training data for the masked acoustic model"""
    with torch.no_grad():
        
        assert(len(spec) == 5), 'dataloader should return (spec_masked, pos_enc, mask_label, attn_mask, spec_stacked)'
        # Unpack and Hack bucket: Bucketing should cause acoustic feature to have shape 1xBxTxD'
        spec_masked = spec[0].squeeze(0)
        pos_enc = spec[1].squeeze(0)
        mask_label = spec[2].squeeze(0)
        attn_mask = spec[3].squeeze(0)
        spec_stacked = spec[4].squeeze(0)

        spec_masked = spec_masked.to(device=device)
        if pos_enc.dim() == 3:
            # pos_enc: (batch_size, seq_len, hidden_size)
            # GPU memory need (batch_size * seq_len * hidden_size)
            pos_enc = torch.FloatTensor(pos_enc).to(device=device)
        elif pos_enc.dim() == 2:
            # pos_enc: (seq_len, hidden_size)
            # GPU memory only need (seq_len * hidden_size) even after expanded
            pos_enc = torch.FloatTensor(pos_enc).to(device=device).expand(spec_masked.size(0), *pos_enc.size())
        mask_label = torch.ByteTensor(mask_label.byte()).to(device=device)
        attn_mask = torch.FloatTensor(attn_mask).to(device=device)
        spec_stacked = spec_stacked.to(device=device)

    return spec_masked, pos_enc, mask_label, attn_mask, spec_stacked # (x, pos_enc, mask_label, attention_mask. y)

#model_name_au = "../GoogleDrive/flm_models/au_aalbert.ckpt"
model_name_au = "result/result_transformer/au_aalbert_3L/states-200000.ckpt"
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
options['ckpt_file'] = model_name_au
args, config = get_upstream_args()
runner = Runner(args, config, None, None, 'result/result_transformer/sample')
runner.set_model()
runner.load_model(model_name_au)
current_transformer = runner.model
current_transformer.eval()
config = {
    'downsample_rate':1,
    'hidden_size':120,
    'mask_proportion':0.15,
    'mask_consecutive_min':1,
    'mask_consecutive_max':1,
    'mask_allow_overlap':True,
    'mask_bucket_ratio':1.2,
    'mask_frequency':0,
    'noise_proportion':0.0
}
#files_path = "/home/mtran/schz_segment/"
files_path = "/shares/perception-working/minh/schz_segment/schz_segment/"
for file in os.listdir(files_path):
    with torch.no_grad():
        
        df = pd.read_csv(os.path.join(files_path, file))
        data = df.values[::3][:,396:413]
        data = torch.from_numpy(data.reshape(1,1,data.shape[0], data.shape[1]))
        batch_is_valid, spec_masked, pos_enc, mask_label, attn_mask, spec_stacked = process_train_MAM_data(data, config)
        spec_masked, pos_enc, mask_label, attn_mask, spec_stacked = process_data([spec_masked, pos_enc, mask_label, attn_mask, spec_stacked])
        loss, pred_spec = current_transformer(spec_masked, pos_enc, mask_label, attn_mask, spec_stacked)
        print("Loss: ",loss)   