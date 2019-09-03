import torch
from torch import nn
import data_deal


class LSTM_Regression(nn.Module):
    def __init__(self, input_size, hidden_size, output_size=1, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, _x):
        x, _ = self.lstm(_x)
        s, b, h = x.shape
        x = x.view(s*b, h)
        x = self.fc(x)
        x = x.view(s, b, -1)
        return x


model = LSTM_Regression(data_deal.DAYS_FOR_TRAIN, 8, output_size=1, num_layers=2).to(data_deal.DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
loss_function = nn.MSELoss()
