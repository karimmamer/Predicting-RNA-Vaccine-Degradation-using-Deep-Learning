import numpy as np
import pandas as pd
import json
import torch
import time
import copy
from torch import nn
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold
import torch.nn.functional as F
import os
import argparse

from cnn import CNN
from cnn_lstm import CNN_LSTM
from cnn_lstm_transformer import CNN_LSTM_Transformer
from dataset import VaxDataset, VaxDatasetPreTrain
from utils import train_model_snapshot, pretrain_model, sn_mcrmse_loss_v2, test, comp_metric

parser = argparse.ArgumentParser(description='single model k-fold training script')
parser.add_argument('-dp','--data_path', help='path to data folder', default='../data', type=str)
parser.add_argument('-T','--temp', help='temperature at which base pairing probability (bpp) matrix is generated', default=37, type=int)
parser.add_argument('-p','--package', help='name of secondary structure prediction package to be used [vienna_2, nupack, rnasoft, rnastructures, contrafold_2, eternafold]', default='eternafold', type=str)
parser.add_argument('-K','--k_folds', help='number of validatikon folds', default=5, type=int)
parser.add_argument('-NS','--n_structures', help='number of secondary structures to be generated for each sequence', default=5, type=int)
parser.add_argument('-a','--arch', help='model architecture to be trained [cnn, cnn_lstm, cnn_lstm_transformer]', default='cnn_lstm', type=str)
args = parser.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#set all seeds
torch.manual_seed(0)
np.random.seed(0)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

#read train data
train_df = pd.read_json(os.path.join(args.data_path, 'train.json'), lines=True)
train_bpps = np.load(os.path.join(args.data_path, 'train_bpps_t_%d_%s.npy'%(args.temp, args.package)))
train_extra_strcutures = np.load(os.path.join(args.data_path, 'train_extra_strcutures_0_to_%d_log_gamma_t_%d_%s.npy'%(args.n_structures-1, args.temp, args.package)))
train_extra_pred_loop = np.load(os.path.join(args.data_path, 'train_extra_pred_loop_0_to_%d_log_gamma_t_%d_%s.npy'%(args.n_structures-1, args.temp, args.package)))

#read test data
test_df = pd.read_json(os.path.join(args.data_path, 'test.json'), lines=True)
test_bpps = np.load(os.path.join(args.data_path, 'test_bpps_t_%d_%s.npy'%(args.temp, args.package)), allow_pickle=True)
test_extra_strcutures = np.load(os.path.join(args.data_path, 'test_extra_strcutures_0_to_%d_log_gamma_t_%d_%s.npy'%(args.n_structures-1, args.temp, args.package)))
test_extra_pred_loop = np.load(os.path.join(args.data_path, 'test_extra_pred_loop_0_to_%d_log_gamma_t_%d_%s.npy'%(args.n_structures-1, args.temp, args.package)))

kf = KFold(n_splits=args.k_folds, random_state=2020, shuffle=True)

if 'transformer' in args.arch:
    lr = 0.0008
    num_epochs = 30
else:
    lr = 0.002
    num_epochs = 20

#self supervised pretraining
print('Pretraining')
dummy_mat = np.zeros((2400, 2))

pretrained_models_arr = []
train_index_arr = []
val_index_arr = []

fold = 0
for train_index, val_index in kf.split(dummy_mat):
    print('Fold %d'%(fold))
    train_index_arr.append(train_index)
    val_index_arr.append(val_index)
    fold += 1
    datasets = {'train': VaxDatasetPreTrain(train_df.iloc[train_index], train_bpps[train_index], train_extra_strcutures[train_index], train_extra_pred_loop[train_index], args.n_structures),
                'val': VaxDatasetPreTrain(train_df.iloc[val_index], train_bpps[val_index], train_extra_strcutures[val_index], train_extra_pred_loop[val_index], args.n_structures),
                'test': VaxDatasetPreTrain(test_df, test_bpps, test_extra_strcutures, test_extra_pred_loop, args.n_structures)}
    
    datasets['train'] = torch.utils.data.ConcatDataset([datasets['train'], datasets['test']])

    dataloaders = {'train': torch.utils.data.DataLoader(datasets['train'], batch_size=16, shuffle=True, num_workers=2),
                   'val': torch.utils.data.DataLoader(datasets['val'], batch_size=128, shuffle=False, num_workers=2)}

    if args.arch == 'cnn':
        model = CNN()
    elif args.arch == 'cnn_lstm_transformer':
        model = CNN_LSTM_Transformer()
    else:
        model = CNN_LSTM()
    
    model = model.to(device)

    dataset_sizes = {x:len(datasets[x]) for x in ['train', 'val']}
    
    model = pretrain_model(model, lr, dataloaders, dataset_sizes, device, num_epochs)
    pretrained_models_arr.append(model)

#supervised fine-tuning
print('Finetuning')
pred = []
target = []
models_arr = []

dummy_mat = np.zeros((2400, 2))

fold = 0
for train_index, val_index in zip(train_index_arr, val_index_arr):
    print('Fold %d'%(fold))
    fold += 1
    datasets = {'train': VaxDataset(train_df, train_bpps, train_extra_strcutures, train_extra_pred_loop, args.n_structures, train_index, True),
                'val': VaxDataset(train_df, train_bpps, train_extra_strcutures, train_extra_pred_loop, args.n_structures, val_index, False)}

    dataloaders = {'train': torch.utils.data.DataLoader(datasets['train'], batch_size=16, shuffle=True, num_workers=2),
                   'val': torch.utils.data.DataLoader(datasets['val'], batch_size=128, shuffle=False, num_workers=2)}

    if args.arch == 'cnn':
        model = CNN()
    elif args.arch == 'cnn_lstm_transformer':
        model = CNN_LSTM_Transformer()
    else:
        model = CNN_LSTM()
    model = model.to(device)
    model.load_state_dict(pretrained_models_arr[fold-1].state_dict())

    criterion = sn_mcrmse_loss_v2
    metric = nn.MSELoss(reduction='none')

    dataset_sizes = {x:len(datasets[x]) for x in ['train', 'val']}
    
    #train a model on this data split using snapshot ensemble
    model_ft_arr, sc, fold_pred, val_target = train_model_snapshot(model, criterion, metric, lr, dataloaders, dataset_sizes, device,
                           num_cycles = 4, num_epochs_per_cycle = num_epochs, n_extra = args.n_structures)
    pred.append(fold_pred)
    target.append(val_target)
    models_arr.extend(model_ft_arr)

#evaluate the error on all folds
pred = np.concatenate(pred)
target = np.concatenate(target)
print('validation mcrmse error:', comp_metric(pred, target))

print('Creating submission')
#predicting on public test set
pub_idx = np.where(test_df['seq_length'] == 107)[0]
pub_test_bpps = [test_bpps[i] for i in pub_idx]
pub_test_dataset = VaxDataset(test_df.iloc[pub_idx], pub_test_bpps, test_extra_strcutures[pub_idx], test_extra_pred_loop[pub_idx], args.n_structures, [], False, True)
pub_test_dataloader = DataLoader(pub_test_dataset, 128, shuffle=False, num_workers=2)
pub_pred = test(models_arr, pub_test_dataloader, device, len(pub_idx)*68, 68, args.n_structures)

#predicting on private test set
pri_idx = np.where(test_df['seq_length'] == 130)[0]
pri_test_bpps = [test_bpps[i] for i in pri_idx]
pri_test_dataset = VaxDataset(test_df.iloc[pri_idx], pri_test_bpps, test_extra_strcutures[pri_idx], test_extra_pred_loop[pri_idx], args.n_structures, [], False, True)
pri_test_dataloader = DataLoader(pri_test_dataset, 128, shuffle=False, num_workers=2)
pri_pred = test(models_arr, pri_test_dataloader, device, len(pri_idx)*91, 91, args.n_structures)

#write predictions in a submission file
sub = pd.read_csv(os.path.join(args.data_path, 'sample_submission.csv'), index_col='id_seqpos')
sub.loc[pub_test_dataset.ids, ['reactivity', 'deg_Mg_pH10', 'deg_Mg_50C']] = pub_pred
sub.loc[pri_test_dataset.ids, ['reactivity', 'deg_Mg_pH10', 'deg_Mg_50C']] = pri_pred
sub.to_csv('%s_with_%s_package_t_%d.csv'%(args.arch, args.package, args.temp))
