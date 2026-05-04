from __future__ import annotations

try:
    import torch
    from torch import nn
except Exception:  # pragma: no cover - lets the API run even before torch is installed
    torch = None
    nn = None


LSTM_INPUT_SIZE = 15


if nn:

    class RenewableLSTM(nn.Module):
        def __init__(self, input_size: int = LSTM_INPUT_SIZE, hidden_size: int = 32) -> None:
            super().__init__()
            self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=1, batch_first=True)
            self.head = nn.Sequential(nn.Linear(hidden_size, 16), nn.ReLU(), nn.Linear(16, 1), nn.Sigmoid())

        def forward(self, x):
            output, _ = self.lstm(x)
            return self.head(output[:, -1, :]).squeeze(-1)

else:
    RenewableLSTM = None
