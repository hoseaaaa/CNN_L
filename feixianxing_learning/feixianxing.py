
#3.快速搭建神经网络实现非线性回归

import torch
import matplotlib.pyplot as plt
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def regression():
    # x = torch.unsqueeze(torch.linspace(-1, 1, 50), dim=1)
    # y = x.pow(2) + torch.rand(x.size())

    x_dict = [0.25, 0.3, 0.375, 0.5, 0.75, 0.04, 0.1, 0.2, 0.275, 0.285, 0.38, 0.4]
    y_dict = [116.508, 112.4, 109.537, 117.107, 128.164, 130.379, 127.385, 114.789, 115.379, 109.306, 114.868, 116.506]

    # Combine features and labels into a TensorDataset
    x = torch.unsqueeze(torch.tensor(x_dict), dim=1)
    y = torch.unsqueeze(torch.tensor(y_dict), dim=1)


    # y = torch.tensor(y_dict)
    
    # torch_dataset = Data.TensorDataset(x_tensor, y_tensor)

    net = torch.nn.Sequential(
        torch.nn.Linear(1, 30),  # 增加第一层的神经元数量
        torch.nn.ReLU(),          # 使用ReLU激活函数
        torch.nn.Linear(30, 20),  # 添加一层隐藏层
        torch.nn.ReLU(),
        torch.nn.Linear(20, 10),  # 添加一层隐藏层
        torch.nn.ReLU(),
        torch.nn.Linear(10, 1)     # 输出层
    )
    plt.ion()
    plt.show()
    # optimizer = torch.optim.Adam(net.parameters(), lr=0.1)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.01, weight_decay=1e-5)  # 降低学习率，添加权重衰减项
    loss_fuc = torch.nn.MSELoss()

    for t in range(18000):
        prediction = net(x)
        loss = loss_fuc(prediction, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if t % 10 == 0:
            plt.cla()
            plt.scatter(x, y)
            plt.scatter(x.data.numpy(), prediction.data.numpy(), color='red', label='Prediction')
            plt.text(0.5, 1.02, 'Loss=%.4f' % loss.item(), transform=plt.gca().transAxes,fontsize=14, color='red', ha='center', va='center')
            # plt.pause(0.1)
            plt.savefig(f'./plot/plot_t_{t}.png')
            print('t:', t,  '|loss:', loss)
    plt.ioff()
    plt.show()


if __name__ == '__main__':
    regression()


# plt.savefig("tmp.png")
