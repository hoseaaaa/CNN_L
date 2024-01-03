import torch
import torch.nn as nn
import torch.optim as optim
import sys

from load_sita import load_sita
import train_model
from coo_dataset import COODataset
from torch.utils.data import TensorDataset
from torch.utils.data import Dataset, DataLoader

from model import Cnn
if __name__ == '__main__':
    #建立数据集
    file_prefix = "../build/coo_dataset/train_dataset"
    coo_dataset = COODataset(file_prefix)
    
    # 从 COODataset 中获取 COO 矩阵和目标值
    COO = [coo_dataset[i][0] for i in range(len(coo_dataset))]
    target = [coo_dataset[i][1] for i in range(len(coo_dataset))]
    #matrix & t
    train_dataset = TensorDataset(torch.stack(COO), torch.stack(target))
    train_data = DataLoader(train_dataset, batch_size=1, shuffle=True)
    #创建网络
    model = Cnn(out_node=1)
    # 损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    load_sita()
    # 模型训练
    train_model.train_model(model, train_data, criterion, optimizer, num_epochs=10)
