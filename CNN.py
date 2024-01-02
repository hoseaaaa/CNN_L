import torch
import torch.nn as nn
import torch.optim as optim

import random

from torch.utils.data import Dataset, DataLoader,random_split
from torch.utils.data import TensorDataset
import matplotlib.pyplot as plt

class Cnn(nn.Module):
    def __init__(self, out_node=1):
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

def load_model(model, filepath):
    # Load the saved model
    model.load_state_dict(torch.load(filepath))
    model.eval()  # Set the model to evaluation mode

def inference(model, val_loader,loss_function):
    # Inference using the model
    return 0


def train_model(model, train_loader, criterion, optimizer, num_epochs=100):
    losses = []
    for epoch in range(num_epochs):
        for _, (coo_matrix, target) in enumerate(train_loader):
            input_data = coo_matrix
            optimizer.zero_grad()
            output = model(input_data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

        losses.append(loss.item())
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}")

        # 保存模型状态
        # torch.save(model.state_dict(), f'AMG_trained_model_epoch_{epoch + 1}.pth')
    # 保存模型状态
    torch.save(model.state_dict(), 'AMG_trained_model.pth')  # 保存模型状态
    #绘制变化曲线
    plt.plot(losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    # 保存绘图
    plt.savefig('training_curve.png')

def validate(model, val_loader, criterion):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for _, (coo_matrix, target) in enumerate(val_loader):
            output = model(coo_matrix)
            loss = criterion(output, target)
            total_loss += loss.item()
    avg_loss = total_loss / len(val_loader)
    return avg_loss

def test(model, test_loader, criterion):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for _, (coo_matrix, target) in enumerate(test_loader):
            output = model(coo_matrix)
            loss = criterion(output, target)
            total_loss += loss.item()
    avg_loss = total_loss / len(test_loader)

    # 如果需要可视化，你可以绘制测试集上的预测结果和实际目标值
    # 这里简单绘制第一个样本的预测结果和目标值
    with torch.no_grad():
        sample_coo, sample_target = test_dataset[0]
        sample_output = model(sample_coo.unsqueeze(0))
    
    plt.figure()
    plt.plot(sample_output.numpy(), label='Predicted')
    plt.plot(sample_target.numpy(), label='Ground Truth')
    plt.legend()
    plt.savefig('training_curve_test.png')

    return avg_loss


if __name__ == '__main__':
    #建立数据集
    file_prefix = "coo_dataset"
    num_samples = 20
    coo_dataset = COODataset(file_prefix, num_samples)

    #划分数据集
    train_size = int(0.7 * len(coo_dataset))
    val_size = int(0.15 * len(coo_dataset))
    test_size = len(coo_dataset) - train_size - val_size

    # 从 COODataset 中获取 COO 矩阵和目标值
    COO = [coo_dataset[i][0] for i in range(len(coo_dataset))]
    target = [coo_dataset[i][1] for i in range(len(coo_dataset))]

    #matrix & t
    torch_dataset = TensorDataset(torch.stack(COO), torch.stack(target))

    train_dataset, val_dataset, test_dataset = random_split(torch_dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=1, shuffle=False)
    test_loader  = DataLoader(test_dataset, batch_size=1, shuffle=False)
    #创建网络
    model = Cnn(out_node=1)
    # 损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
        
    # 模型训练
    # train_model(model, train_loader, criterion, optimizer, num_epochs=10)

    loaded_net = Cnn()
    load_model(loaded_net, 'AMG_trained_model.pth')
    # 在验证集上评估模型
    val_loss = validate(model, val_loader, criterion)
    print(f"Average Loss on val_loss Set: {val_loss}")

    # 在测试集上评估模型
    test_loss = test(model, test_loader, criterion)
    print(f"Average Loss on Test Set: {test_loss}")
