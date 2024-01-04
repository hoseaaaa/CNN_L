import torch
import torch.nn as nn

import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

def train_model(model, train_loader, criterion, optimizer, num_epochs=100):
    losses = []
    for epoch in range(num_epochs):
        for _, (coo_matrix, target,data_x1,data_x2,data_x3,data_x4) in enumerate(train_loader):
            input_data = coo_matrix
            optimizer.zero_grad()
            output = model(input_data,data_x1,data_x2,data_x3,data_x4)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

        losses.append(loss.item())
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}")

        # 保存模型状态
        # torch.save(model.state_dict(), f'AMG_trained_model_epoch_{epoch + 1}.pth')
    # 保存模型状态
    torch.save(model.state_dict(), '../build/AMG_trained_model.pth')  # 保存模型状态
    #绘制变化曲线
    plt.plot(losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    # 保存绘图
    plt.savefig('../build/training_curve.png')
