import numpy as np
import torch
import torch.nn as nn
from .base_model import BaseVolatilityModel


class BaseNNVolatilityModel(nn.Module, BaseVolatilityModel):
    def __init__(self):
        nn.Module.__init__(self)
        BaseVolatilityModel.__init__(self)
        self.device = torch.device(
            "mps" if torch.backends.mps.is_available() else "cpu"
        )

    def to_device(self):
        self.to(self.device)
        for param in self.parameters():
            param.data = param.data.to(self.device)

    def fit(self, X, y, returns, start_index, end_index):
        self.to_device()
        X_tensor = torch.FloatTensor(X[start_index:end_index]).to(self.device)
        y_tensor = torch.FloatTensor(y[start_index:end_index]).to(self.device)

        optimizer = torch.optim.Adam(self.parameters())
        criterion = nn.MSELoss()

        for epoch in range(5):  # Increased number of epochs
            optimizer.zero_grad()
            output = self(X_tensor)
            loss = criterion(output, y_tensor)
            loss.backward()
            optimizer.step()

        return self

    def predict(self, X, returns):
        self.to_device()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)  # Add time dimension
            predictions = self(X_tensor)
            predictions = predictions.cpu().numpy()
        return predictions


class RNNVolatilityModel(BaseNNVolatilityModel):
    def __init__(self, input_size, hidden_size, num_layers=1):
        super(RNNVolatilityModel, self).__init__()
        self.rnn = nn.RNN(
            input_size, hidden_size, num_layers, batch_first=False, bias=False
        )
        self.fc = nn.Linear(hidden_size, 1)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.to_device()

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, 1, self.hidden_size).to(self.device)
        out, _ = self.rnn(x.unsqueeze(1), h0)
        out = self.fc(out[:, -1, :])
        return out.squeeze()


class GRUVolatilityModel(BaseNNVolatilityModel):
    def __init__(self, input_size, hidden_size, num_layers=1):
        super(GRUVolatilityModel, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=False)
        self.fc = nn.Linear(hidden_size, 1)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.to_device()

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, 1, self.hidden_size).to(self.device)
        out, _ = self.gru(x.unsqueeze(1), h0)
        out = self.fc(out[:, -1, :])
        return out.squeeze()
