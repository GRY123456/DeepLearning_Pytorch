import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import data_deal


class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, 5)  # [BATCH, 1, 28, 28] -> [BATCH, 10, 24, 24]
        self.conv2 = nn.Conv2d(10, 20, 3)  # [BATCH, 10, 12, 12] -> [BATCH, 20, 10, 10]
        self.fc1 = nn.Linear(20*10*10, 500)  # [BATCH, 20*10*10] -> [BATCH, 500]
        self.fc2 = nn.Linear(500, 10)  # [BATCH, 500] -> [BATCH, 10]

    def forward(self, x):
        in_size = x.size(0)  # x.size() = [BATCH, 1, 28, 28]
        out = F.relu(self.conv1(x))  # [BATCH, 1, 28, 28] -> [BATCH, 10, 24, 24]
        out = F.max_pool2d(out, 2, 2)  # [BATCH, 10, 24, 24] -> [BATCH, 10, 12, 12]
        out = F.relu(self.conv2(out))  # [BATCH, 10, 12, 12] -> [BATCH, 20, 10, 10]
        out = out.view(in_size, -1)  # out.size() = [BATCH, 20*10*10]
        out = F.relu(self.fc1(out))  # [BATCH, 20*10*10] -> [BATCH, 500]
        out = self.fc2(out)  # [BATCH, 500] -> [BATCH, 10]
        out = F.log_softmax(out, dim=1)
        return out


model = ConvNet().to(data_deal.DEVICE)
optimizer = optim.Adam(model.parameters())
