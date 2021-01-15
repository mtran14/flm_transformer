import os
import numpy as np
from torch.utils.data.dataset import Dataset
import re 
import pandas as pd

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
    
class SchizophreniaMMDataset(Dataset):
    def __init__(self, file_paths, labels, max_len=100):
        self.labels = labels
        
        self.X = []
        pad = lambda a, i: a[0: i,:] if a.shape[0] > i else np.concatenate((a, np.zeros((i - a.shape[0],a.shape[1]))), axis=0)
        for i in range(len(file_paths)):
            file = file_paths[i]
            label = labels[i]
            try:
                current_data = np.load(file[3:])
            except:
                current_data = np.load(file)
            if(len(current_data) <= 2):
                continue
            
            facial_landmarks = current_data[:,1:137]
            gaze_pose = current_data[:,137:148]
            aus = current_data[:,148:165]
            gpau = np.concatenate((aus, gaze_pose), axis=1)
            data_dict = {"flm":pad(facial_landmarks, max_len),\
                         "gp":pad(gaze_pose, max_len), \
                         "au":pad(aus, max_len),\
                         "gpau":pad(gpau, max_len)}            
            self.X.append([data_dict, label, "pseudo-string"])
                
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, index):
        return self.X[index]
    
class AvecDataset(Dataset):
    def __init__(self, file_paths, scores, max_len=500):
        assert len(file_paths) == len(scores)
        self.X = []
        for i in range(len(scores)):
            try:
                current_data = np.load(file_paths[i]) #n_frames x n_features
            except:
                continue
            current_score = scores[i]
            participant_id = int(re.findall(r'\d+', file_paths[i])[0])
            index = 0
            
            while(index + max_len <= current_data.shape[0]):
                current_chunk = current_data[index:index+max_len,:]
                facial_landmarks = current_chunk[:,0:136]
                gaze_pose = current_chunk[:,136:147]
                aus = current_chunk[:,147:164]
                data_dict = {"flm":facial_landmarks,\
                             "gp":gaze_pose, \
                             "au":aus}                
                self.X.append([data_dict, current_score, participant_id])
                index += max_len
                
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, index):
        return self.X[index]   
    
class AvecDatasetFull(Dataset):
    def __init__(self, file_paths, scores, max_len=5000):
        assert len(file_paths) == len(scores)
        self.X = []
        pad = lambda a, i: a[0: i,:] if a.shape[0] > i else np.concatenate((a, np.zeros((i - a.shape[0],a.shape[1]))), axis=0)
        for i in range(len(scores)):
            try:
                current_data = np.load(file_paths[i]) #n_frames x n_features
            except:
                continue
            current_score = scores[i]
            participant_id = int(re.findall(r'\d+', file_paths[i])[0])
            
            facial_landmarks = current_data[:,0:136]
            gaze_pose = current_data[:,136:147]
            aus = current_data[:,147:164]
            gpau = np.concatenate((aus, gaze_pose), axis=1)
            data_dict = {"flm":pad(facial_landmarks,max_len),\
                         "gp":pad(gaze_pose,max_len), \
                         "au":pad(aus,max_len), \
                         "gpau":pad(gpau, max_len)}                
            self.X.append([data_dict, current_score, participant_id]) 
                
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, index):
        return self.X[index]    
    
class MosiDataset(Dataset):
    def __init__(self, user_ids, score_file, file_paths, subset, max_len=50):
        #path2np is a dict contain multiple sources; subset tell which source
        self.X = []
        score_data = pd.read_csv(score_file, header=None).values
        pad = lambda a, i: a[0: i,:] if a.shape[0] > i else np.concatenate((a, np.zeros((i - a.shape[0],a.shape[1]))), axis=0)
        for i in range(score_data.shape[0]):
            keep = False
            current_row = score_data[i]
            for user_id in user_ids:
                if(user_id in current_row[0]):
                    keep = True
            if(not keep):
                continue
            file_root = current_row[0].split(".")[0]
            output = []
            
            file_path = os.path.join(file_paths, file_root+".npy")
            current_score = current_row[1]
            try:
                current_data = np.load(file_path) #n_frames x n_features (all 3 modalities)
            except:
                continue
            facial_landmarks = current_data[:,0:136]
            gaze_pose = current_data[:,136:147]
            aus = current_data[:,147:164]
            data_dict = {"flm":pad(facial_landmarks, max_len),\
                         "gp":pad(gaze_pose, max_len), \
                         "au":pad(aus, max_len)}
            #subset_data = np.concatenate([data_dict[x] for x in subset])
            #pad_data = pad(subset_data, max_len)  
            #except:
                #continue
            
            self.X.append([data_dict, current_score, "pseudo-string"])
                
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, index):
        return self.X[index]  