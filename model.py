import torch
import torch.nn as nn

class BiLSTM(nn.Module):
    def __init__(self, args):
        super(BiLSTM, self).__init__()
        
        self.lstm = nn.LSTM(args.input_size, args.num_units, args.num_layers, batch_first=False, bidirectional=True)
        self.fc = nn.Linear(args.num_units*2, args.num_classes)
        self.loss= nn.LogSoftmax()
        self.num_layers = args.num_layers
        self.num_units = args.num_units

    def forward(self, x):
        out, h = self.lstm(x)
        out = self.fc(out)
        return out

    def export(self):
        pass
