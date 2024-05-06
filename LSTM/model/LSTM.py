import torch
import torch.nn as nn

class BiLSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_size=256, dropout_rate=0.25):
        super(BiLSTMModel, self).__init__()
        self.lstm1 = nn.LSTM(input_size=input_dim, hidden_size=204, num_layers=1, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout_rate)
        self.lstm2 = nn.LSTM(input_size=hidden_size*2, hidden_size=int(hidden_size/2), num_layers=1, batch_first=True, bidirectional=True)
        self.dense = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x, _ = self.lstm1(x)
        x = self.dropout(x)
        x, _ = self.lstm2(x)
        # 取最后一个时间步的输出
        x = self.dense(x[:, -1, :])
        return x
    
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_size=256, dropout_rate=0.25):
        super(LSTMModel, self).__init__()
        self.lstm1 = nn.LSTM(input_size=input_dim, hidden_size=hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout_rate)
        self.lstm2 = nn.LSTM(input_size=hidden_size, hidden_size=int(hidden_size/2), batch_first=True)
        self.dense = nn.Linear(int(hidden_size/2), 1)

    def forward(self, x):
        x, _ = self.lstm1(x)
        x = self.dropout(x)
        x, _ = self.lstm2(x)
        x = self.dense(x[:, -1, :])  # 只取最后一个时间步的输出
        return x