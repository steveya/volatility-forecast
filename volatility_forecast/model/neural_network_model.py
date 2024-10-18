import numpy as np
import torch
import torch.nn as nn
from .base_model import BaseVolatilityModel


class BaseNNVolatilityModel(nn.Module, BaseVolatilityModel):
    def __init__(self, max_grad_norm=1):
        nn.Module.__init__(self)
        BaseVolatilityModel.__init__(self)
        self.device = torch.device(
            "mps" if torch.backends.mps.is_available() else "cpu"
        )
        self.max_grad_norm = max_grad_norm

    def to_device(self):
        self.to(self.device)
        for param in self.parameters():
            param.data = param.data.to(self.device)

    def fit(self, X, y, **kwargs):
        start_index = kwargs.pop("start_index", 0)
        end_index = kwargs.pop("end_index", len(X))
        epochs = kwargs.pop("epochs", 50)
        verbose = kwargs.pop("verbose", True)  # Add a verbose option
        learning_rate = kwargs.pop("learning_rate", 0.01)
        self.to_device()

        X_tensor = torch.FloatTensor(X[start_index:end_index]).to(self.device)
        y_tensor = torch.FloatTensor(y[start_index:end_index]).to(self.device)

        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

        criterion = nn.MSELoss()

        for epoch in range(epochs):  # Increased number of epochs
            self.train()
            optimizer.zero_grad()
            output = self(X_tensor).reshape(-1, 1)
            loss = criterion(output, y_tensor)

            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), max_norm=self.max_grad_norm)
            optimizer.step()
            scheduler.step()

            if verbose:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.6f}")

        return self

    def predict(self, X, **kwargs):
        self.to_device()
        self.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)  # Add time dimension
            predictions = self(X_tensor)
            predictions = predictions.cpu().numpy()
        return predictions

    def print_parameters(self):
        print(f"Model Parameters for {self.__class__.__name__}:")
        for name, param in self.named_parameters():
            if param.requires_grad:
                print(f"{name}:")
                print(f"  Shape: {param.shape}")
                print(f"  Data (first few values): {param.data.flatten()[:5]}")
                print(
                    f"  Gradient (first few values): {param.grad.flatten()[:5] if param.grad is not None else 'None'}"
                )
                print()

    def print_hidden_states(self, X):
        self.eval()
        with torch.no_grad():
            X = torch.FloatTensor(X).unsqueeze(0)  # Add batch dimension
            hidden = self.init_hidden(1)  # Initialize hidden state
            output, hidden_states = self(X, hidden)

            print(f"Hidden States Shape: {hidden_states.shape}")
            print("Hidden States Sequence:")
            for t in range(hidden_states.shape[1]):
                print(f"Time step {t}:")
                for layer in range(self.num_layers):
                    print(
                        f"  Layer {layer}: {hidden_states[layer, 0, :5].numpy()}"
                    )  # Print first 5 values

            return hidden_states.numpy()


class RNNVolatilityModel(BaseNNVolatilityModel):
    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers=1,
        dropout=0,
        bias=False,
        nonlinearity="tanh",
        max_grad_norm=1,
    ):
        super(RNNVolatilityModel, self).__init__(max_grad_norm)
        self.rnn = nn.RNN(
            input_size,
            hidden_size,
            num_layers,
            batch_first=False,
            dropout=dropout,
            bias=bias,
            nonlinearity=nonlinearity,
        )
        self.fc = nn.Linear(hidden_size, 1)
        self.bn_input = nn.BatchNorm1d(input_size)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.to_device()

    def init_hidden(self):
        return torch.zeros(self.num_layers, 1, self.hidden_size).to(self.device)

    def forward(self, x):
        x = self.bn_input(x)
        h0 = self.init_hidden()
        out, _ = self.rnn(x.unsqueeze(1), h0)
        out = self.fc(out[:, -1, :])
        out = torch.exp(out).squeeze()
        return out


class GRUVolatilityModel(BaseNNVolatilityModel):
    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers=1,
        dropout=0,
        bias=False,
        max_grad_norm=1,
    ):
        super(GRUVolatilityModel, self).__init__(max_grad_norm)
        self.gru = nn.GRU(
            input_size,
            hidden_size,
            num_layers,
            batch_first=False,
            dropout=dropout,
            bias=bias,
        )
        self.fc = nn.Linear(hidden_size, 1)
        self.bn_input = nn.BatchNorm1d(input_size)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.to_device()

    def init_hidden(self):
        return torch.zeros(self.num_layers, 1, self.hidden_size).to(self.device)

    def forward(self, x):
        x = self.bn_input(x)
        h0 = self.init_hidden()
        out, _ = self.gru(x.unsqueeze(1), h0)
        out = self.fc(out[:, -1, :])
        out = torch.exp(out).squeeze()
        return out


class RNNSTESModel(BaseNNVolatilityModel):
    def __init__(self, input_size, hidden_size, num_layers=1, dropout=0):
        super(RNNSTESModel, self).__init__()
        self.rnn = nn.RNN(
            input_size,
            hidden_size,
            num_layers,
            batch_first=False,
            dropout=dropout,
        )

        self.fc = nn.Linear(hidden_size, 1)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.to_device()

    def init_hidden(self):
        return torch.zeros(self.num_layers, 1, self.hidden_size).to(self.device)

    def forward(self, x, returns):
        h0 = self.init_hidden()
        out, _ = self.rnn(x.unsqueeze(1), h0)
        alphas = torch.sigmoid(self.fc(out[:, -1, :]))
        returns_squared = returns**2

        sigma_t_minus_1 = returns_squared[0] * torch.ones(1).to(self.device)
        sigma_t = sigma_t_minus_1
        output = []
        for t in range(x.shape[0]):
            sigma_t = alphas[t] * returns_squared[t] + (1 - alphas[t]) * sigma_t_minus_1
            output.append(sigma_t)
            sigma_t_minus_1 = sigma_t
        output = torch.stack(output)

        return output

    def fit(self, X, y, **kwargs):
        returns = kwargs.pop("returns", None)
        start_index = kwargs.pop("start_index", 0)
        end_index = kwargs.pop("end_index", len(X))
        epochs = kwargs.pop("epochs", 50)

        assert returns is not None, "Returns must be provided"

        X_tensor = torch.FloatTensor(X[start_index:end_index]).to(self.device)
        y_tensor = torch.FloatTensor(y[start_index:end_index]).to(self.device)
        returns_tensor = torch.FloatTensor(returns[start_index:end_index]).to(
            self.device
        )

        optimizer = torch.optim.Adam(self.parameters())
        criterion = nn.MSELoss()

        for epoch in range(epochs):
            self.train()
            optimizer.zero_grad()
            output = self(X_tensor, returns_tensor)
            loss = criterion(output, y_tensor)
            loss.backward()
            optimizer.step()

        return self

    def predict(self, X, **kwargs):
        returns = kwargs.pop("returns", None)

        self.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            returns_tensor = torch.FloatTensor(returns).to(self.device)
            predictions = self(X_tensor, returns_tensor)
            predictions = predictions.cpu().numpy()
        return predictions
