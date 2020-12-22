import os
import numpy as np
from torch.utils.data.dataset import Dataset

class SchizophreniaDataset(Dataset):
    def __init__(self, file_paths, chunk_size, labels):
        self.labels = labels
        self.chunk_size = chunk_size
        
        self.X = []
        for i in range(len(file_paths)):
            file = file_paths[i]
            label = labels[i]
            current_data = np.load(file[3:])
            start_index = 0
            while(start_index + chunk_size <= current_data.shape[0]):
                self.X.append([current_data[start_index:start_index+chunk_size,:], label, file])
                start_index += chunk_size
                
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, index):
        return self.X[index]
          
          
class SchizophreniaSegmentDataset(Dataset):
    def __init__(self, file_paths, labels, max_len=500):
        self.labels = labels
        
        self.X = []
        pad = lambda a, i: a[0: i,:] if a.shape[0] > i else np.concatenate((a, np.zeros((i - a.shape[0],a.shape[1]))), axis=0)
        for i in range(len(file_paths)):
            file = file_paths[i]
            label = labels[i]
            current_data = np.load(file[3:])
            if(len(current_data) <= 2):
                continue
            pad_data = pad(current_data, max_len)
            self.X.append([pad_data, label, "pseudo-string"])
                
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, index):
        return self.X[index]