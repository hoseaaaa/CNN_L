import torch
import torch.nn as nn
import torch.utils.data as Data
import matplotlib.pyplot as plt
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

class create_model(nn.Module):
    def __init__(self):
        super(create_model, self).__init__()
        self.model = nn.Sequential(
        nn.Linear(3, 400),
        nn.ReLU(),
        nn.BatchNorm1d(400),  # Batch Normalization
        nn.Linear(400, 200),
        nn.ReLU(),
        nn.Linear(200, 100),
        nn.ReLU(),
        nn.Linear(100, 50),
        nn.ReLU(),
        nn.BatchNorm1d(50),  # Batch Normalization
        nn.Linear(50, 50),
        nn.ReLU(),
        nn.Linear(50, 1)
    )

    def forward(self, x):
        return self.model(x)


def create_dataset():
    # Define the dataset
    # x = torch.unsqueeze(torch.linspace(-20, 30, 2000), dim=1)
    # y = 3 * x.pow(2) + torch.rand(x.size())
    # torch_dataset = Data.TensorDataset(x, y)
    
    # x_dict = [0.25, 0.3, 0.375, 0.5, 0.75, 0.04, 0.1, 0.2, 0.275, 0.285, 0.38, 0.4]
    # y_dict = [116.508, 112.4, 109.537, 117.107, 128.164, 130.379, 127.385, 114.789, 115.379, 109.306, 114.868, 116.506]
    
    x1_dict = [0.25, 0.3, 0.375, 0.5, 0.75, 0.04, 0.1, 0.2, 0.275, 0.285, 0.38, 0.4, 0.41, 0.42, 0.43, 0.45, 0.46, 0.47, 0.48, 0.49, 0.52, 0.54, 0.6, 0.7,0.65,0.8,0.9,
            0.25, 0.3, 0.375, 0.5, 0.75, 0.04, 0.1, 0.2, 0.275, 0.285, 0.38, 0.4, 0.41, 0.42, 0.43, 0.45, 0.46, 0.47, 0.48, 0.49, 0.52, 0.54, 0.6,0.7,0.65,0.8,0.9,  
            0.25, 0.3, 0.375, 0.5, 0.75, 0.04, 0.1, 0.2, 0.275, 0.285, 0.38, 0.4, 0.41, 0.42, 0.43, 0.45, 0.46, 0.47, 0.48, 0.49, 0.52, 0.54, 0.6,0.7,0.65,0.8,0.9,
            0.25, 0.3, 0.375, 0.5, 0.75, 0.04, 0.1, 0.2, 0.275, 0.285, 0.38, 0.4, 0.41, 0.42, 0.43, 0.45, 0.46, 0.47, 0.48, 0.49, 0.52, 0.54, 0.6,0.7,0.65,0.8,0.9]

    # x2_dict = [4125504, 4125504, 4125504, 4125504, 4125504, 4125504, 4125504, 4125504, 4125504, 4125504, 4125504, 4125504, 4125504, 4125504, 4125504, 4125504, 4125504, 4125504, 4125504, 4125504, 4125504, 4125504, 4125504, 4125504,4125504,4125504,4125504,
            #    21469972, 21469972, 21469972, 21469972, 21469972, 21469972, 21469972, 21469972, 21469972, 21469972, 21469972, 21469972, 21469972, 21469972, 21469972, 21469972, 21469972, 21469972, 21469972, 21469972, 21469972, 21469972, 21469972, 21469972, 21469972, 21469972, 21469972,
            #    39072262, 39072262, 39072262, 39072262, 39072262, 39072262, 39072262, 39072262, 39072262, 39072262, 39072262, 39072262, 39072262, 39072262, 39072262, 39072262, 39072262, 39072262, 39072262, 39072262, 39072262, 39072262, 39072262, 39072262,39072262,39072262,39072262,
            #    51173098,51173098,51173098,51173098,51173098,51173098,51173098,51173098,51173098,51173098,51173098,51173098,51173098,51173098,51173098,51173098,51173098,51173098,51173098,51173098,51173098,51173098,51173098,51173098,51173098,51173098,51173098]
    # x3_dict = [834252,834252,834252,834252,834252,834252,834252,834252,834252,834252,834252,834252,834252,834252,834252,834252,834252,834252,834252,834252,834252,834252,834252,834252,834252,834252,834252,834252,
            #    4974178,4974178,4974178,4974178,4974178,4974178,4974178,4974178,4974178,4974178,4974178,4974178,4974178,4974178,4974178,4974178,4974178,4974178,4974178,4974178,4974178,4974178,4974178,4974178,4974178,4974178,4974178,
            #    8988782,8988782,8988782,8988782,8988782,8988782,8988782,8988782,8988782,8988782,8988782,8988782,8988782,8988782,8988782,8988782,8988782,8988782,8988782,8988782,8988782,8988782,8988782,8988782,8988782,8988782,8988782,
            #    11777726,11777726,11777726,11777726,11777726,11777726,11777726,11777726,11777726,11777726,11777726,11777726,11777726,11777726,11777726,11777726,11777726,11777726,11777726,11777726,11777726,11777726,11777726,11777726,11777726,11777726,11777726]
    x2_dict = [
        15.23269875210551,15.23269875210551,15.23269875210551,15.23269875210551,15.23269875210551,15.23269875210551,15.23269875210551,15.23269875210551,15.23269875210551,15.23269875210551,15.23269875210551,15.23269875210551,15.23269875210551,15.23269875210551,15.23269875210551,15.23269875210551,15.23269875210551,15.23269875210551,15.23269875210551,15.23269875210551,15.23269875210551,15.23269875210551,15.23269875210551,15.23269875210551,15.23269875210551,15.23269875210551,15.23269875210551,
        16.882165865708796,16.882165865708796,16.882165865708796,16.882165865708796,16.882165865708796,16.882165865708796,16.882165865708796,16.882165865708796,16.882165865708796,16.882165865708796,16.882165865708796,16.882165865708796,16.882165865708796,16.882165865708796,16.882165865708796,16.882165865708796,16.882165865708796,16.882165865708796,16.882165865708796,16.882165865708796,16.882165865708796,16.882165865708796,16.882165865708796,16.882165865708796,16.882165865708796,16.882165865708796,16.882165865708796,
        17.48092336143929,17.48092336143929,17.48092336143929,17.48092336143929,17.48092336143929,17.48092336143929,17.48092336143929,17.48092336143929,17.48092336143929,17.48092336143929,17.48092336143929,17.48092336143929,17.48092336143929,17.48092336143929,17.48092336143929,17.48092336143929,17.48092336143929,17.48092336143929,17.48092336143929,17.48092336143929,17.48092336143929,17.48092336143929,17.48092336143929,17.48092336143929,17.48092336143929,17.48092336143929,17.48092336143929,
        17.750724522235686,17.750724522235686,17.750724522235686,17.750724522235686,17.750724522235686,17.750724522235686,17.750724522235686,17.750724522235686,17.750724522235686,17.750724522235686,17.750724522235686,17.750724522235686,17.750724522235686,17.750724522235686,17.750724522235686,17.750724522235686,17.750724522235686,17.750724522235686,17.750724522235686,17.750724522235686,17.750724522235686,17.750724522235686,17.750724522235686,17.750724522235686,17.750724522235686,17.750724522235686,17.750724522235686
    ] #lnx2
    x3_dict = [
        13.634290793973648,13.634290793973648,13.634290793973648,13.634290793973648,13.634290793973648,13.634290793973648,13.634290793973648,13.634290793973648,13.634290793973648,13.634290793973648,13.634290793973648,13.634290793973648,13.634290793973648,13.634290793973648,13.634290793973648,13.634290793973648,13.634290793973648,13.634290793973648,13.634290793973648,13.634290793973648,13.634290793973648,13.634290793973648,13.634290793973648,13.634290793973648,13.634290793973648,13.634290793973648,13.634290793973648,
        15.419770688792836,15.419770688792836,15.419770688792836,15.419770688792836,15.419770688792836,15.419770688792836,15.419770688792836,15.419770688792836,15.419770688792836,15.419770688792836,15.419770688792836,15.419770688792836,15.419770688792836,15.419770688792836,15.419770688792836,15.419770688792836,15.419770688792836,15.419770688792836,15.419770688792836,15.419770688792836,15.419770688792836,15.419770688792836,15.419770688792836,15.419770688792836,15.419770688792836,15.419770688792836,15.419770688792836,
        16.011487913398067,16.011487913398067,16.011487913398067,16.011487913398067,16.011487913398067,16.011487913398067,16.011487913398067,16.011487913398067,16.011487913398067,16.011487913398067,16.011487913398067,16.011487913398067,16.011487913398067,16.011487913398067,16.011487913398067,16.011487913398067,16.011487913398067,16.011487913398067,16.011487913398067,16.011487913398067,16.011487913398067,16.011487913398067,16.011487913398067,16.011487913398067,16.011487913398067,16.011487913398067,16.011487913398067,
        16.281720678504044,16.281720678504044,16.281720678504044,16.281720678504044,16.281720678504044,16.281720678504044,16.281720678504044,16.281720678504044,16.281720678504044,16.281720678504044,16.281720678504044,16.281720678504044,16.281720678504044,16.281720678504044,16.281720678504044,16.281720678504044,16.281720678504044,16.281720678504044,16.281720678504044,16.281720678504044,16.281720678504044,16.281720678504044,16.281720678504044,16.281720678504044,16.281720678504044,16.281720678504044,16.281720678504044
    ] #lnx3

    y_dict =[2.996, 2.925, 2.785, 3.14, 2.732, 3.105, 3.1, 2.994, 3.003, 2.909, 2.656, 6.997, 2.948, 2.942, 2.709, 2.755, 2.778, 2.764, 2.881, 2.864, 2.765, 2.942, 3.157, 2.654,2.72,2.68,2.788,
            29.665, 20.528, 22.918, 22.064, 33.344, 40.896, 28.018, 35.475, 21.383, 24.133, 24.346, 25.162, 22.463, 22.444, 27.292, 21.253, 21.476, 27.447, 25.247, 21.102, 22.319, 24.646, 21.638, 22.174,21.961,23.2,32.968,
            57.048, 54.732, 53.781, 57.674, 65.929, 72.752, 64.068, 53.42, 56.189, 59.388, 57.791, 54.098, 54.669, 55.572, 53.694, 59.278, 56.652, 55.851, 57.967, 56.959, 53.932, 60.112, 59.897, 64.104,58.933,68.135,73.396,
            80.506, 75.07, 76.83, 81.082, 89.71, 92.896, 83.919, 80.666, 73.901, 77.196, 74.412, 76.348, 74.655, 85.416, 81.52, 78.47, 77.103, 77.215, 83.862, 76.245, 81.199, 74.5, 75.702, 85.35, 82.191, 91.12, 102.486]


    x_tensor = torch.tensor(list(zip(x1_dict, x2_dict,x3_dict)))
    y_tensor = torch.unsqueeze(torch.tensor(y_dict), dim=1)

    # Combine features and labels into a TensorDataset
    
    # x_tensor = torch.unsqueeze(torch.tensor(x_dict), dim=1)
    # y_tensor = torch.unsqueeze(torch.tensor(y_dict), dim=1)

    torch_dataset = Data.TensorDataset(x_tensor, y_tensor)


    # Define the splitting ratios
    train_ratio = 0.7
    val_ratio = 0.15
    test_ratio = 0.15

    # Calculate the sizes for each split
    dataset_size = len(torch_dataset)
    train_size = int(train_ratio * dataset_size)
    val_size = int(val_ratio * dataset_size)
    test_size = dataset_size - train_size - val_size

    # Use torch.utils.data.random_split to split the dataset
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        torch_dataset, [train_size, val_size, test_size]
    )

    return train_dataset, val_dataset, test_dataset

def create_data_loaders(train_dataset, val_dataset, test_dataset, batch_size=50):
    # Define the data loaders
    train_loader = Data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
    )
    val_loader = Data.DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
    )
    test_loader = Data.DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
    )

    return train_loader, val_loader, test_loader

def train_model(model, train_loader, optimizer, loss_function, num_epochs=5000, plot_frequency=5):
    plt.ion()
    plt.show()

    for epoch in range(num_epochs):
        for step, (batch_x, batch_y) in enumerate(train_loader):
            prediction = model(batch_x)
            loss = loss_function(prediction, batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if epoch % plot_frequency == 0:
            # plt.cla()
            # plt.scatter(batch_x.numpy(), batch_y.numpy(), label='Ground Truth')
            # plt.scatter(batch_x.numpy(), prediction.detach().numpy(), color='red', label='Prediction')
            # plt.text(0.5, 0, 'Loss=%.4f' % loss.item(), fontdict={'size': 20, 'color': 'red'})
            # plt.legend()
            # plt.pause(0.1)
            # plt.savefig(f'./plot2/plot_t_{epoch}.png')
            print('Epoch:', epoch, ' |Step:', step, ' |loss:', loss)
    torch.save(model.state_dict(), 'trained_model.pth')  # 保存模型状态
    # 训练完毕后关闭图表
    # plt.ioff()
    # plt.show()
def test_model(model, test_loader, loss_function):
    # Test the model
    with torch.no_grad():
        test_loss = 0.0
        for test_step, (test_x, test_y) in enumerate(test_loader):
            test_prediction = model(test_x)
            test_loss += loss_function(test_prediction, test_y)
        print('Test Loss:', test_loss / (test_step + 1))

def inference(model, val_loader,loss_function):
    # Inference using the model
    all_outputs = []
    with torch.no_grad():
        val_loss = 0.0
        model.eval()  # Set the model to evaluation mode
        for val_step, (val_x, val_y) in enumerate(val_loader):
            output_y = model(val_x)
            val_loss += loss_function(output_y, val_y)
            print('val Loss:', val_loss / (val_step + 1))
            all_outputs.append(output_y)
        model.train()  # Set the model back to training mode
    return torch.cat(all_outputs, dim=0)

def load_model(model, filepath):
    # Load the saved model
    model.load_state_dict(torch.load(filepath))
    model.eval()  # Set the model to evaluation mode


def find_min_x1(model, x2_fixed, x1_values):
    # Fix x2 and iterate over different x1 values
    x_fixed = torch.tensor([[0.0, x2_fixed]])

    min_y = float('inf')
    min_x1 = None

    # Iterate over x1 values
    for x1 in x1_values:
        x_fixed[0, 0] = x1  # Update x1 with the current value
        y_prediction = model(x_fixed)
        
        if y_prediction.item() < min_y:
            min_y = y_prediction.item()
            min_x1 = x1

    return min_x1, min_y


if __name__ == '__main__':
    # Create and load datasets
    train_dataset, val_dataset, test_dataset = create_dataset()

    # Create data loaders
    Batch_size = 50
    train_loader, val_loader, test_loader = create_data_loaders(train_dataset, val_dataset, test_dataset, Batch_size)

    # Create the neural network model
    net = create_model()

    # Create the optimizer and loss function
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    # change optimeter

    # optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    loss_function = torch.nn.MSELoss()

    # Train the model
    train_model(net, train_loader, optimizer, loss_function)

    # Test the model
    loaded_net = create_model()

    load_model(loaded_net, 'trained_model.pth')

    test_model(loaded_net, test_loader, loss_function)

    # Inference using the loaded model
    # input_data = torch.unsqueeze(torch.linspace(-20, 30, 200), dim=1)
    output = inference(loaded_net, val_loader,loss_function)
    print ("interface model ")

    # Visualize the results
    # plt.scatter(train_dataset[:][0].data.numpy(), train_dataset[:][1].data.numpy(), lw=1, label='Training Data')
    # plt.scatter(val_dataset[:][0].data.numpy(), output.data.numpy(), 'r-', lw=2, label='Loaded Model Prediction')
    # plt.legend()
    # plt.show()

    # plt.scatter(train_dataset[:][0].data.numpy(), train_dataset[:][1].data.numpy(), lw=1, label='Training Data')
    # plt.scatter(val_dataset[:][0].data.numpy(), val_dataset[:][1].data.numpy(), label='Ground Truth (Validation)', alpha=0.5)
    # plt.scatter(val_dataset[:][0].data.numpy(), output.data.numpy(), color='red', label='Loaded Model Prediction')
    # plt.legend()
    # plt.savefig(f'./plot2/plot_t_interface model.png')
    # plt.show()


    # # 定义采样的 x 范围
    # x_min, x_max = 0.0, 1.0
    # num_samples = 1000

    # # 在指定范围内对 x 进行均匀采样
    # x_samples = torch.linspace(x_min, x_max, num_samples).view(-1, 1)

    # # 使用训练好的模型预测对应的 y
    # loaded_net.eval()
    # y_predictions = loaded_net(x_samples)

    # # 找到预测的 y 的最小值及其索引
    # min_y, min_y_index = torch.min(y_predictions, dim=0)

    # # 得到对应于最小 y 的 x 值
    # x_at_min_y = x_samples[min_y_index]

    # print(f"The minimum y value is {min_y.item()} at x = {x_at_min_y.item()}")

    x1_values = torch.linspace(0.1, 0.95, 1000)
    x2_fixed_value = 18.010326081432268

    loaded_net.eval()
    min_x1, min_y = find_min_x1(loaded_net, x2_fixed_value, x1_values)
    print(f"The minimum y value is {min_y} at x1 = {min_x1} for x2 = {x2_fixed_value}")