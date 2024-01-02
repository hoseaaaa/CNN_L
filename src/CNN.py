import torch
import torch.nn as nn
import torch.optim as optim

import random

from torch.utils.data import Dataset, DataLoader,random_split
from torch.utils.data import TensorDataset
import matplotlib.pyplot as plt

from load_sita import *
import model

def load_model(model, filepath):
    # Load the saved model
    model.load_state_dict(torch.load(filepath))
    model.eval()  # Set the model to evaluation mode

def inference(model, val_loader,loss_function):
    # Inference using the model
    return 0



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


def load_function(file_prefix, num_samples):
    m_data = []
    
    for idx in range(1, num_samples + 1):
        filename = f"./{file_prefix}/{idx}.txt"
        
        with open(filename, 'r') as file:
            # 读取每个文件的第一行第一个数据，并转换为浮点数
            first_data = float(file.readline().split()[0])
            m_data.append(first_data)

    # 将数据存储到 m.txt
    with open("m.txt", 'w') as m_file:
        for data in m_data:
            m_file.write(f"{data}\n")


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
