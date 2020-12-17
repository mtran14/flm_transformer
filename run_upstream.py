# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ run_upstream.py ]
#   Synopsis     [ scripts for running the pre-training of transformer models ]
#   Author       [ Andy T. Liu (Andi611) ]
#   Copyright    [ Copyleft(c), Speech Lab, NTU, Taiwan ]
"""*********************************************************************************************"""


###############
# IMPORTATION #
###############
import os
import glob
import yaml
import torch
import random
import argparse
import numpy as np
from shutil import copyfile
from argparse import Namespace
from dataloader import get_Dataloader, get_online_Dataloader
from utility.helper import parse_prune_heads


#################
# PATH HANDLING #
#################
import sys
S3PRL_PATH = os.getcwd() # or set this to your own path that points to the S3PRL repo
if S3PRL_PATH not in sys.path:
    sys.path.append(S3PRL_PATH)


######################
# UPSTREAM ARGUMENTS #
######################
def get_upstream_args():
    
    parser = argparse.ArgumentParser(description='Argument Parser for Upstream Models of the S3PLR project.')

    # required, set either (--run and --config) or (--resume)
    parser.add_argument('--run', default=None, choices=['transformer', 'apc'], help='Select pre-training task. \
                        For the transformer models, which type of pre-training (mockingjay, tera, aalbert, etc) \
                        is determined by config file.')
    parser.add_argument('--config', default=None, type=str, help='Path to experiment config.')
    parser.add_argument('--resume', default=None, help='Specify the upstream checkpoint path to resume training')

    # ckpt and logging
    parser.add_argument('--name', default=None, type=str, help='Name for logging.')
    parser.add_argument('--ckpdir', default='', type=str, help='Path to store checkpoint result, if empty then default is used.')
    parser.add_argument('--seed', default=1337, type=int, help='Random seed for reproducable results.')
    
    # Options
    parser.add_argument('--test', default='', type=str, help='Input path to the saved model ckpt for testing.')
    parser.add_argument('--cpu', action='store_true', help='Disable GPU training.')
    parser.add_argument('--multi_gpu', action='store_true', help='Enable Multi-GPU training.')
    parser.add_argument('--test_reconstruct', action='store_true', help='Test reconstruction capability')
    parser.add_argument('--online_config', default=None, help='Explicitly specify the config of on-the-fly feature extraction')
    parser.add_argument('--kaldi_data', action='store_true', help='Whether to use the Kaldi dataset')

    # parse
    args = parser.parse_args()
    if args.resume is None:
        assert args.run is not None and args.config is not None, '`--run` and `--config` must be given if `--resume` is not provided'
        setattr(args, 'gpu', not args.cpu)
        config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)
        parse_prune_heads(config)
        if args.online_config is not None:
            online_config = yaml.load(open(args.online_config, 'r'), Loader=yaml.Loader)
            config['online'] = online_config
    else:
        if os.path.isdir(args.resume):
            ckpts = glob.glob(f'{args.resume}/*.ckpt')
            assert len(ckpts) > 0
            ckpts = sorted(ckpts, key=lambda pth: int(pth.split('-')[-1].split('.')[0]))
            resume_ckpt = ckpts[-1]
        else:
            resume_ckpt = args.resume

        def update_args(old, new):
            old_dict = vars(old)
            new_dict = vars(new)
            old_dict.update(new_dict)
            return Namespace(**old_dict)

        ckpt = torch.load(resume_ckpt, map_location='cpu')
        args = update_args(args, ckpt['Settings']['Paras'])
        config = ckpt['Settings']['Config']
        setattr(args, 'resume', resume_ckpt)        
    
    return args, config


##################
# GET DATALOADER #
##################
def get_dataloader(args, config):
    
    if not os.path.exists(config['dataloader']['data_path']):
        raise RuntimeError('[run_upstream] - Data path not valid:', config['dataloader']['data_path'])
    print('[run_upstream] - Loading input data: ' + str(config['dataloader']['train_set']) + ' from ' + config['dataloader']['data_path'])
    print('[run_upstream] - getting train dataloader...')

    # select mode
    load = 'duo' if bool(config['runner']['duo_feature']) else 'kaldi' if args.kaldi_data else 'flm'

    # print path info
    if load == 'duo': 
        print('[run_upstream] - Loading duo data: ' + str(config['dataloader']['train_set']) + ' from ' + config['dataloader']['target_path'])
    elif load == 'kaldi':
        print('[run_upstream] - Loading Kaldi data: ' + str(config['dataloader']['data_path']) + ' from these sets ' + str(config['dataloader']['train_set']))
    elif load == 'acoustic' and 'online' in config:
        print('[run_upstream] - Using online data from root: ' + str(config['online']['libri_root']))
    elif load == 'acoustic':
        print('[run_upstream] - Loading data: ' + str(config['dataloader']['data_path']) + ' from these sets ' + str(config['dataloader']['train_set']))
    elif load == 'flm':
        print('[run_upstream] - Loading data: ' + str(config['dataloader']['data_path']) + ' from these sets ' + str(config['dataloader']['train_set']))    

    dataloader = get_Dataloader(split='train', load=load, use_gpu=args.gpu, 
                                run_mam=True, mam_config=config['transformer'], **config['dataloader'], **config)

    return dataloader


###################
# RUN TRANSFORMER #
###################
def run_transformer(args, config):
    from transformer.runner import Runner

    # mkdir
    if args.ckpdir == '':
        if args.name is None: args.name = 'run_' + str(random.randint(0, 999))
        ckpdir = os.path.join('result/result_transformer/', args.name)
    else:
        ckpdir = args.ckpdir
    if not os.path.exists(ckpdir):
        os.makedirs(ckpdir)
    copyfile(args.config, os.path.join(ckpdir, args.config.split('/')[-1]))
    if args.online_config is not None:
        copyfile(args.online_config, os.path.join(ckpdir, args.online_config.split('/')[-1]))

    # get dataloader
    dataloader = get_dataloader(args, config)

    # train
    runner = Runner(args, config, dataloader, ckpdir)
    runner.set_model()
    runner.train()


####################
# TEST TRANSFORMER #
####################
def test_transformer(args, input_dim):
    from transformer.nn_transformer import TRANSFORMER
    options = {'ckpt_file'     : args.test,
               'load_pretrain' : 'True',
               'no_grad'       : 'True',
               'dropout'       : 'default',
               'spec_aug'      : 'False',
               'spec_aug_prev' : 'True',
               'weighted_sum'  : 'False',
               'select_layer'  : -1,
    }
    upstream_model = TRANSFORMER(options, input_dim)
    print('[upstream runner] - successfully loaded, valid checkpoint: ', args.test)
    return upstream_model


###########
# RUN APC #
###########
def run_apc(seed):
    from rnn.runner import Runner
    runner = Runner(seed, train=True)
    runner.train()


########
# MAIN #
########
def main():
    
    # get arguments
    args, config = get_upstream_args()
    
    # Fix seed and make backends deterministic
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

    # Train Transformer
    if args.run == 'transformer':
        if args.test != '':
            test_transformer(args, config['transformer']['input_dim'])
        else:
            run_transformer(args, config)

    elif args.run == 'apc':
        if args.test != '':
            raise NotImplementedError
        else:
            run_apc(args.seed)


if __name__ == '__main__':
    main()