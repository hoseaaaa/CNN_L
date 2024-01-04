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

        self.t_file_prefix =  file_prefix.replace("coo_dataset", "target")
        self.scalar_file_prefix = file_prefix.replace("coo_dataset", "x_dataset")
        self.target_file = f"./{self.t_file_prefix}/target.txt"
        self.scalar_file_x1 = f"./{self.scalar_file_prefix}/lnn.txt"
        self.scalar_file_x2 = f"./{self.scalar_file_prefix}/lnnnz.txt"
        self.scalar_file_x3 = f"./{self.scalar_file_prefix}/relax.txt"
        self.scalar_file_x4 = f"./{self.scalar_file_prefix}/sita.txt"

        self.file_list = self.get_file_list()
        self.num_samples = len(self.file_list)

        self.targets = self.load_scalar(self.target_file)

        self.scalar_x1 = self.load_scalar(self.scalar_file_x1)
        self.scalar_x2 = self.load_scalar(self.scalar_file_x2)
        self.scalar_x3 = self.load_scalar(self.scalar_file_x3)
        self.scalar_x4 = self.load_scalar(self.scalar_file_x4)

    def get_file_list(self):
        # Get a list of all .txt files in the specified directory
        file_list = [filename for filename in os.listdir(self.file_prefix) if os.path.isfile(os.path.join(self.file_prefix, filename)) and filename.endswith(".txt")]
        return file_list

    def load_scalar(self,filename):
        data = []
        with open(filename, 'r') as file:
            for line in file:
                data.extend(map(float, line.split()))
        return data

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        filename = f"./{self.file_prefix}/{idx + 1}.txt"
        coo_matrix = self.load_coo_matrix(filename)
        target = torch.tensor(self.targets[idx], dtype=torch.float32).unsqueeze(0)
        x1 = torch.tensor(self.scalar_x1[idx], dtype=torch.float32).unsqueeze(0)
        x2 = torch.tensor(self.scalar_x2[idx], dtype=torch.float32).unsqueeze(0)
        x3 = torch.tensor(self.scalar_x3[idx], dtype=torch.float32).unsqueeze(0)
        x4 = torch.tensor(self.scalar_x4[idx], dtype=torch.float32).unsqueeze(0)
        return coo_matrix, target, x1, x2, x3, x4

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
