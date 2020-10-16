'''
Refrence script by @its7171: https://www.kaggle.com/its7171/how-to-generate-augmentation-data
'''
import numpy as np
import pandas as pd
import json
import torch
import time
import copy
from arnie.bpps import bpps
from multiprocessing import Pool
from tqdm import tqdm
from arnie.mea.mea import MEA
import os
import sys
import argparse

def calc_bpp(seq): 
    return bpps(seq, package=args.package, T=args.temp)

def calc_struct(inputs):
    seq, bpp = inputs
    struct_arr = []
    for log_gamma in range(0, args.n_structures):
        mea_mdl = MEA(bpp,gamma=10**log_gamma)
        struct_arr.append(mea_mdl.structure)
    return struct_arr

def get_predicted_loop_type(id, sequence, structure, debug=False):
    structure_fixed = structure.replace('.','0').replace('(','1').replace(')','2')
    pid = os.getpid()
    tmp_in_file = f'{args.data_path}/tmp_files/{id}_{structure_fixed}_{pid}.dbn'
    tmp_out_file = f'{id}_{structure_fixed}_{pid}.st'
    os.system('echo %s > %s'%(sequence, tmp_in_file))
    os.system('echo "%s" >> %s'%(structure, tmp_in_file))
    os.system('perl bpRNA/bpRNA.pl %s'%(tmp_in_file))
    result = [l.strip('\n') for l in open(tmp_out_file)]
    if debug:
        print(sequence)
        print(structure)
        print(result[5])
    else:
        os.system('rm %s %s'%(tmp_out_file, tmp_in_file))
    return result[5]

def calc_pred_loop(inputs):
    id, seq, struct_arr = inputs
    pred_loop_arr = []
    for i in range(0, args.n_structures):
        pred_loop = get_predicted_loop_type(id, seq, struct_arr[i])
        pred_loop_arr.append(pred_loop)
    return pred_loop_arr

parser = argparse.ArgumentParser(description='Extra data (secondary structres and pridected loop types) generation script')
parser.add_argument('-dp','--data_path', help='path to data folder', default='../data', type=str)
parser.add_argument('-T','--temp', help='temperature at which base pairing probability (bpp) matrix is generated', default=37, type=int)
parser.add_argument('-p','--package', help='name of secondary structure prediction package to be used [vienna_2, nupack, rnasoft, rnastructures, contrafold_2, eternafold]', default='eternafold', type=str)
parser.add_argument('-NT','--n_threads', help='number of parallel threads', default=16, type=int)
parser.add_argument('-NS','--n_structures', help='number of secondary structures to be generated for each sequence', default=5, type=int)
args = parser.parse_args()

#create tmp folder (if it doesn't exist)
if not os.path.exists(os.path.join(args.data_path, 'tmp_files')):
    os.mkdir(os.path.join(args.data_path, 'tmp_files'))

df_arr = [pd.read_json(os.path.join(args.data_path, 'train.json'), lines=True),
          pd.read_json(os.path.join(args.data_path, 'test.json'), lines=True)]

split_arr = ['train', 'test']

#generate for both train and test data
for df, split in zip(df_arr, split_arr):
    print('%s split'%(split))
    
    #step 1: generate bpp for each sequence and save them
    print('generating bpp')
    p = Pool(processes=args.n_threads)

    bpps_arr = []

    for bpp in tqdm(p.imap(calc_bpp, df['sequence'].values),total=df.shape[0]):
        bpps_arr.append(bpp)

    np.save(os.path.join(args.data_path, '%s_bpps_t_%d_%s'%(split, args.temp, args.package)), bpps_arr)

    #step 2: generate "n_structures" secondary structures per single sequence and save them
    print('generating secondary structures')
    p = Pool(processes=args.n_threads)

    extra_strcutures_arr = []

    seq_bpp_pair_arr = [(df['sequence'].iloc[i], bpps_arr[i]) for i in range(df.shape[0])]

    for extra_strcutures in tqdm(p.imap(calc_struct, seq_bpp_pair_arr),total=df.shape[0]):
        extra_strcutures_arr.append(extra_strcutures)
    extra_strcutures_arr = np.array(extra_strcutures_arr)

    np.save(os.path.join(args.data_path, '%s_extra_strcutures_0_to_%d_log_gamma_t_%d_%s'%(split, args.n_structures-1, args.temp, args.package)), extra_strcutures_arr)

    #step 3: generate predicted loop type for each generated secondary
    print('generating loop type')
    p = Pool(processes=args.n_threads)

    extra_pred_loops_arr = []

    id_seq_struct_triplet_arr = [(df['id'].iloc[i], df['sequence'].iloc[i], extra_strcutures_arr[i]) for i in range(df.shape[0])]

    for pred_loops in tqdm(p.imap(calc_pred_loop, id_seq_struct_triplet_arr), total=df.shape[0]):
        extra_pred_loops_arr.append(pred_loops)

    extra_pred_loops_arr = np.array(extra_pred_loops_arr)

    np.save(os.path.join(args.data_path, '%s_extra_pred_loop_0_to_%d_log_gamma_t_%d_%s'%(split, args.n_structures-1, args.temp, args.package)), extra_pred_loops_arr)
