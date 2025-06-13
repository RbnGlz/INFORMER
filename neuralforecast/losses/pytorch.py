import torch
from torch import nn

class MAE(nn.Module):
    outputsize_multiplier = 1
    def forward(self, y_pred, y_true):
        return torch.mean(torch.abs(y_pred - y_true))
