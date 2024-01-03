import torch
import torch.nn as nn
import torch.optim as optim

import random
import os

from torch.utils.data import Dataset, DataLoader,random_split
from torch.utils.data import TensorDataset

class COODataset(Dataset):
    def __init__(self, file_prefix):
        self.file_prefix = file_prefix
        self.b_file_prefix =  file_prefix.replace("coo_dataset", "target")
        self.file_list = self.get_file_list()
        self.num_samples = len(self.file_list)
        # self.num_samples = num_samples
        self.target_filename = f"./{self.b_file_prefix}/target.txt"
        self.targets = self.load_targets()

    def get_file_list(self):
        # Get a list of all .txt files in the specified directory
        file_list = [filename for filename in os.listdir(self.file_prefix) if os.path.isfile(os.path.join(self.file_prefix, filename)) and filename.endswith(".txt")]
        return file_list

    def load_targets(self):
        targets = []
        with open(self.target_filename, 'r') as target_file:
            for line in target_file:
                targets.extend(map(float, line.split()))
        return targets

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        filename = f"./{self.file_prefix}/{idx + 1}.txt"
        coo_matrix = self.load_coo_matrix(filename)
        target = torch.tensor(self.targets[idx], dtype=torch.float32).unsqueeze(0)
        return coo_matrix, target

    def load_coo_matrix(self, filename):
        with open(filename, 'r') as file:
            # Read COO matrix from file
            rows, cols, nnz = map(int, file.readline().split())
            row_indices = list(map(int, file.readline().split()))
            col_indices = list(map(int, file.readline().split()))
            values = list(map(float, file.readline().split()))

        # Create dense matrix from COO format
        dense_matrix = torch.zeros((rows, cols), dtype=torch.float32)
        for i in range(nnz):
            dense_matrix[row_indices[i], col_indices[i]] = values[i]

        return dense_matrix.unsqueeze(0)
