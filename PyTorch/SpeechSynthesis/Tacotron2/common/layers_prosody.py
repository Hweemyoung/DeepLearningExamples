import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from common.layers import *

class Conv2DNorm(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size=(3,3), stride=2, padding=1, padding_mode='zero',
                 w_init_gain='linear'):
        super(Conv2DNorm, self).__init__()
        self.Conv2D_layer = nn.Conv2d(in_dim,
                                      out_dim,
                                      kernel_size=kernel_size,
                                      stride=stride,
                                      padding=padding,
                                      padding_mode=padding_mode
                                      )
        nn.init.xavier_uniform_(
            self.Conv2D_layer.weight,
            gain=nn.init.calculate_gain(w_init_gain)
        )

    def forward(self, x):
        return self.Conv2D_layer(x)


class ReferenceEncoder(nn.Module):
    def __init__(self, dim_ref, len_ref, in_sizes=(1, 32, 32, 64, 64, 128), out_sizes=(32, 32, 64, 64, 128, 128), rnn_mode='GRU', rnn_units=128, dim_prosody=128):
        super(ReferenceEncoder, self).__init__()
        # 6-Layer Strided Conv2D w/ BatchNorm
        assert(len(in_sizes) == len(out_sizes))
        self.layers = nn.ModuleList()
        for (in_size, out_size) in zip(in_sizes, out_sizes):
            self.layers.append(
                Conv2DNorm(in_size, out_size)
            )
            self.layers.append(
                nn.ReLU()
            )
            self.layers.append(
                nn.BatchNorm2d(num_features=out_size)
            )
        # 128-unit GRU
        self.rnn = nn.GRU(input_size=out_sizes[-1]*int(dim_r / 2**(len(self.layers)/3)), hidden_size=rnn_units)
        self.Linear = LinearNorm(in_dim=rnn_units, out_dim=dim_prosody)

    def forward(self, x):
        print(x.size())
        for layer in self.layers:
            x = layer(x)
        print(x.size())
        x = x.flatten(1, 2)
        print(x.size())
        x = x.transpose(1, 2)
        print(x.size())
        x, _ = self.rnn(x)
        print(x.size())
        x = self.Linear(x)
        # get final output
        x = x[:, -1, :]
        print(x.size())
        print(x)
        x = torch.tanh(x)
        return x


dim_r = 128
L_r = 256
refenc = ReferenceEncoder(dim_r,L_r)
refinput = torch.rand(size=[10, 1, dim_r, L_r])
refenc(refinput)