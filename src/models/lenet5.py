import torch.nn as nn
import torch.nn.functional as F
import math
import torch
class LeNet5(nn.Module):
    """
    LeNet-5 模型定义
    """
    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, padding=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(6, 16, kernel_size=5),
        )
        self.fc1 = nn.Sequential(
            nn.Linear(16 * 5 * 5, 120),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(120, 84),
        )
        self.fc3 = nn.Sequential(
            nn.Linear(84, 10),
        )

    def forward(self, x):
        x = F.max_pool2d(torch.tanh(self.conv1(x)), (2, 2))
        x = F.dropout(x, p=0.3, training=self.training)
        x = F.max_pool2d(torch.tanh(self.conv2(x)), (2, 2))
        x = F.dropout(x, p=0.3, training=self.training)
        x = x.view(-1, self.num_flat_features(x))
        x = torch.tanh(self.fc1(x))
        x = F.dropout(x, p=0.3, training=self.training)
        x = torch.tanh(self.fc2(x))
        x = F.dropout(x, p=0.3, training=self.training)
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

def weight_init(m):
    """
    权重初始化函数
    """
    if isinstance(m, nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()
