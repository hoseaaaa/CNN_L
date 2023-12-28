import torch
import torch.nn as nn
import torch.optim as optim

import random

from torch.utils.data import Dataset, DataLoader

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
    denst_matrix_1 = generate_dense_matrix()
    denst_matrix_2 = generate_dense_matrix()
    denst_matrix_3 = generate_dense_matrix()
    denst_matrix_4 = generate_dense_matrix()
    denst_matrix_5 = generate_dense_matrix()
    
    input_data = torch.stack([denst_matrix_1, denst_matrix_2,denst_matrix_3,denst_matrix_4,denst_matrix_5]).unsqueeze(1)  # 添加通道维度
    target = torch.tensor([36.44444321, 12.213124, 32.543534, 76.2344, 89.1242363], dtype=torch.float32)
    target = target.unsqueeze(1)

    #创建网络
    model = Cnn(out_node=1)
    # 损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    # 模型训练
    for epoch in range(100):
        optimizer.zero_grad()
        output = model(input_data)  # 注意：将输入张量的维度增加到 (1, input_size)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch+1}/{100}, Loss: {loss.item()}")

