import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import f1_score

class ConvBNReLU_0(nn.Module):
    def __init__(self, in_channels=1, out_channels=32, kernel_size=9, stride=1, padding=0):
        super(ConvBNReLU_0, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x
    
class LFEM_i(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2, padding=0):
        super(LFEM_i, self).__init__()

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu2 = nn.ReLU()
        self.maxpool = nn.MaxPool1d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.maxpool(x)
        return x

class GRUDense(nn.Module):
    def __init__(self, dropout_prob=0.2, input_size=1, hidden_size=30):
        super(GRUDense, self).__init__()
        self.dropout = nn.Dropout(p=dropout_prob)
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.dense = nn.Linear(hidden_size, 4)

    def forward(self, x):
        x = self.dropout(x)
        x, _ = self.gru(x)
        x = x[:, -1, :]
        x = self.dense(x)
        return x
    
def compute_f1_loss(model: nn, X_samples: np.ndarray, y_target: np.ndarray):
    '''
    model: shold return array with length 4 for each row
    '''
    y = model(X_samples)
    _, y = torch.max(y, 1)
    y, y_target = y.cpu(), y_target.cpu()
    return f1_score(y_target, y, average='micro')