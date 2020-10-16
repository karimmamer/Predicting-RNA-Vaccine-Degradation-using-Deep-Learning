import numpy as np
import pandas as pd
from torch.utils.data import Dataset

def one_hot(categories, string):
    encoding = np.zeros((len(string), len(categories)))
    for idx, char in enumerate(string):
        encoding[idx, categories.index(char)] = 1
    return encoding

def featurize(seq, struct, pred_loop):
    sequence = one_hot(list('ACGU'), seq)
    structure = one_hot(list('.()'), struct)
    loop_type = one_hot(list('BEHIMSX'), pred_loop)
    features = np.hstack([sequence, structure, loop_type])
    return features 

def get_neighbour_list(s, max_len=False):
    '''
    for each nt, find the index of the other nt to which is connected in the secondary structure 
    '''
    if max_len:
        neighbour_list = [130 for _ in range(130)]
    else:
        neighbour_list = [len(s) for _ in range(len(s))]
    q = []
    
    for i in range(len(s)):
        if s[i] == '(':
            q.append(i)
        elif s[i] == ')':
            j = q.pop(-1)
            neighbour_list[i] = j
            neighbour_list[j] = i
        
    neighbour_list = np.array(neighbour_list).astype(np.long)
    
    return neighbour_list

class VaxDataset(Dataset):
    def __init__(self, df, bpps, extra_structures, extra_pred_loops, n_extra, sample_idx, train, test=False):
        self.df = df if test else df.iloc[sample_idx]
        self.bpps = bpps  if test else bpps[sample_idx]
        self.extra_structures = extra_structures if test else extra_structures[sample_idx]
        self.extra_pred_loops = extra_pred_loops if test else extra_pred_loops[sample_idx]
        self.test = test
        self.train = train
        self.n_extra = n_extra
        self.sample_idx = set(list(sample_idx))
        self.features = []
        self.targets = []
        self.masks = []
        self.errors = []
        self.snr = []
        self.ids = []
        self.neighbour_lists = []
        
        self.load_data()
        
    def create_feature(self, seq, struct, pred_loop, bpp):
        graph_features = featurize(seq, struct, pred_loop).transpose((1,0))

        graph_features  = np.concatenate([graph_features, bpp.sum(1).reshape(1,-1)], 0)

        self.features.append(graph_features)

        graph_neighbour_list = get_neighbour_list(struct)
        self.neighbour_lists.append(graph_neighbour_list)
    
    def create_sample(self, records, bpp, extra_struct, extra_pred_loop):
        seq = records['sequence']
        for j in range(len(extra_struct)):
            struct = extra_struct[j]
            pred_loop = extra_pred_loop[j]
            self.create_feature(seq, struct, pred_loop, bpp)

        graph_mask = np.zeros((records['seq_length'],), dtype = np.bool)
        graph_mask[:records['seq_scored']] = True
        self.masks.append(graph_mask)

        for char_i in range(records['seq_scored']):
            self.ids.append('%s_%d' % (records['id'], char_i))

        if not self.test:
            graph_targets = np.stack([records['reactivity'], records['deg_Mg_pH10'], records['deg_Mg_50C'], records['deg_pH10'], records['deg_50C']], axis=1)
            self.targets.append(graph_targets)

            graph_errors = np.stack([records['reactivity_error'], records['deg_error_Mg_pH10'], records['deg_error_Mg_50C'], records['deg_error_pH10'], records['deg_error_50C']], axis=1)
            self.errors.append(graph_errors)

            self.snr.append(records['signal_to_noise'])
        
    def load_data(self):
        for i in range(self.df.shape[0]):
            records = self.df.iloc[i]
            bpp = self.bpps[i]
            extra_struct = self.extra_structures[i]
            extra_pred_loop = self.extra_pred_loops[i]
            if (self.test == False) and (self.train == False) and (records["SN_filter"] == 0.0):
                    continue
            self.create_sample(records, bpp, extra_struct, extra_pred_loop)
                    
    def __len__(self):
        if self.train:
            return len(self.masks)
        return len(self.features)
    
    def __getitem__(self, index):
        if self.test:
            return self.features[index].astype(np.float32), self.masks[index//self.n_extra], self.neighbour_lists[index].astype(np.long)
        else:
            if self.train:
                feat_index = index*self.n_extra + np.random.randint(0, self.n_extra)
                gt_index = index
            else:
                feat_index = index
                gt_index = index//self.n_extra
            
            return self.features[feat_index].astype(np.float32), self.targets[gt_index].astype(np.float32), self.masks[gt_index], self.neighbour_lists[feat_index].astype(np.long), self.errors[gt_index].astype(np.float32), self.snr[gt_index]

class VaxDatasetPreTrain(Dataset):
    def __init__(self, df, bpps, extra_structures, extra_pred_loops, n_extra, nof_droped_nt = 30):
        self.df = df
        self.bpps = bpps
        self.extra_structures = extra_structures
        self.extra_pred_loops = extra_pred_loops
        self.n_extra = n_extra
        self.nof_droped_nt = nof_droped_nt
        self.features = []
        self.neighbour_lists = []
        self.nts = []
        self.load_data()
        
    def create_feature(self, seq, struct, pred_loop, bpp):
        graph_features = featurize(seq, struct, pred_loop).transpose((1,0))

        graph_features = np.concatenate([graph_features, bpp.sum(1).reshape(1,-1)], 0)
        graph_features = np.concatenate([graph_features, np.zeros_like(graph_features)], 1)[:,:130]

        self.features.append(graph_features)

        graph_neighbour_list = get_neighbour_list(struct, True)
        self.neighbour_lists.append(graph_neighbour_list)
        
        nt = np.zeros((130,), dtype = np.long)
        for i, char in enumerate(seq):
            nt[i] = list('ACGU').index(char)
        self.nts.append(nt)
    
    def create_sample(self, records, bpp, extra_struct, extra_pred_loop):
        seq = records['sequence']
        for j in range(self.n_extra):
            struct = extra_struct[j]
            pred_loop = extra_pred_loop[j]
            self.create_feature(seq, struct, pred_loop, bpp)
        
    def load_data(self):
        for i in range(self.df.shape[0]):
            records = self.df.iloc[i]
            bpp = self.bpps[i]
            extra_struct = self.extra_structures[i]
            extra_pred_loop = self.extra_pred_loops[i]
            self.create_sample(records, bpp, extra_struct, extra_pred_loop)
                    
    def __len__(self):
        return self.df.shape[0]

    def mask_nt(self, x, idx, gt):
        rng = np.arange(107)
        np.random.shuffle(rng)
        rng = rng[:self.nof_droped_nt]
        x = np.copy(x)
        x[:, rng] = 0
        mask = np.zeros_like(gt)
        mask[rng] = 1
        idx[rng] = idx.max()
        return x, idx, gt, mask
    
    def __getitem__(self, index):
        index = index*self.n_extra + np.random.randint(0, self.n_extra)
        return self.mask_nt(self.features[index].astype(np.float32), self.neighbour_lists[index].astype(np.long), self.nts[index])
