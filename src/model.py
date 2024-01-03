import torch
import torch.nn as nn

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
        # 新增两个全连接层的输入节点数
        # self.node_sita = 1   
        # self.node_relax = 1 
        # self.node_ln_n = 1   
        # self.node_ln_nnz = 1  

        self.fc = nn.Sequential(
            nn.Linear(10, 200),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(200, out_node)
        )

    def forward(self,x):
    # def forward(self,x,sita,relax,ln_n,ln_nnz):
        # x = self.embedding(x)
        out = self.conv(x)
        out = self.adaptive_pool(out)
        out = out.view(out.size(0), -1)
        # #添加新节点
        # m_data = torch.tensor(m, dtype=torch.float32).unsqueeze(0)
        # n_data = torch.tensor(n, dtype=torch.float32).unsqueeze(0)
        # b_data = torch.tensor(b, dtype=torch.float32).unsqueeze(0)
        # out = torch.cat((out, m_data, n_data, b_data), dim=1)
        out = self.fc(out)
        return out
    