import torch
import torch.nn as nn

class Cnn(nn.Module):
    def __init__(self, out_node=1):
        super(Cnn, self).__init__()
        # self.embedding = nn.EmbeddingBag.from_pretrained(30, 10, sparse=True)
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Additional convolutional layers
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),

        )

        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.Sequential(
            nn.Linear(516, 400),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(400, 300),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(300, 200),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(200, out_node)
        )

    def forward(self, x, x1, x2, x3, x4):
        out = self.conv(x)
        out = self.adaptive_pool(out)
        out = out.view(out.size(0), -1)
        out = torch.cat([out, x1, x2, x3, x4], dim=1)
        out = self.fc(out)
        return out
    