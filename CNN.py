import torch
import torch.nn as nn
import torch.optim as optim

import random

from torch.utils.data import Dataset, DataLoader
from torch.utils.data import TensorDataset
import matplotlib.pyplot as plt

class Cnn(nn.Module):
    def __init__(self, out_node):
        super(Cnn, self).__init__()
        # self.embedding = nn.EmbeddingBag.from_pretrained(30, 10, sparse=True)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=10, kernel_size=3, stride=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=1),
            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=3, stride=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=1),
        )

        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.Sequential(
            nn.Linear(10, 200),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(200, out_node)
        )

    def forward(self, x):
        # x = self.embedding(x)
        out = self.conv(x)
        out = self.adaptive_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out
    

class COODataset(Dataset):
    def __init__(self, file_prefix, num_samples):
        self.file_prefix = file_prefix
        self.num_samples = num_samples
        self.target_filename = f"./target.txt"
        self.targets = self.load_targets()
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

def generate_dense_matrix(density=0.4):
    # 创建一个28x28的零矩阵
    dense_matrix = torch.zeros((28, 28), dtype=torch.float32)
    
    # 计算需要设置为非零值的元素数量
    num_non_zero_elements = int(density * 28 * 28)
    
    # 生成随机位置并设置为非零值
    for _ in range(num_non_zero_elements):
        i, j = random.randint(0, 27), random.randint(0, 27)
        dense_matrix[i, j] = random.uniform(1, 10)
    
    return dense_matrix


if __name__ == '__main__':
    #建立数据集
    file_prefix = "coo_dataset"
    num_samples = 100
    
    coo_dataset = COODataset(file_prefix, num_samples)
    # torch_dataset = TensorDataset(coo_dataset , coo_dataset.targets)

    # data_loader = DataLoader(coo_dataset, batch_size=1, shuffle=True)
    # 从 COODataset 中获取 COO 矩阵和目标值
    COO = [coo_dataset[i][0] for i in range(len(coo_dataset))]
    target = [coo_dataset[i][1] for i in range(len(coo_dataset))]
    # torch_dataset = TensorDataset(*coo_dataset)

    # 将 COO 矩阵和目标值绑定在一起
    # torch_dataset = TensorDataset(*COO, *target)
    torch_dataset = TensorDataset(torch.stack(COO), torch.stack(target))
    data_loader = DataLoader(torch_dataset, batch_size=1, shuffle=True)

    #创建网络
    model = Cnn(out_node=1)
    # 损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
        
    # 模型训练
    losses = []
    
    for epoch in range(10000):
        for _, (coo_matrix, target) in enumerate(data_loader):
            input_data = coo_matrix
            optimizer.zero_grad()
            output = model(input_data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
        losses.append(loss.item())
        print(f"Epoch {epoch + 1}/{100}, Loss: {loss.item()}")

    # 绘制变化曲线
    plt.plot(losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # 保存绘图
    plt.savefig('training_curve.png')