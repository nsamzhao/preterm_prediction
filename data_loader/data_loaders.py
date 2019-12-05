import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import torch
from torch.utils.data import TensorDataset
from base import BaseDataLoader

class AllGenesDataLoader(BaseDataLoader):
    def __init__(self, data_dir, batch_size, shuffle = True, validation_split = 0.0, num_workers = 1, training = True):
        self.data_dir = data_dir
        self.X = torch.load(os.path.join(self.data_dir, "X_train.pt"))
        self.y = torch.load(os.path.join(self.data_dir, "y_train_onehot.pt"))
        self.dataset = TensorDataset(self.X, self.y)

        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)

class SelectedGenesDataLoader(BaseDataLoader):
    def __init__(self, data_dir, csv_filename, labels_filename, batch_size, shuffle = True, validation_split = 0.0, num_workers = 1, training = True):
        self.data_dir = data_dir

        data_df = pd.read_csv(os.path.join(self.data_dir, csv_filename), header = None)
        data = np.float32(data_df.values)
        labels_df = pd.read_csv(os.path.join(self.data_dir, labels_filename), header = None)
        labels = np.float32(labels_df.values)  
        
        # Convert labels to one-hot for BCE loss
        enc = OneHotEncoder()
        enc.fit(labels)
        labels_onehot = np.float32(enc.transform(labels).toarray())
        
        assert data.shape[1] == 100     # ensure only 100 genes
        assert data.shape[0] == labels_onehot.shape[0]

        self.X = torch.from_numpy(data) 
        self.y = torch.from_numpy(labels_onehot)
        self.dataset = TensorDataset(self.X, self.y)

        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)

