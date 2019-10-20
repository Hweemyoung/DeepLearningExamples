import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from common.layers import ConvNorm, LinearNorm

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

class MatMul:
    def __init__(self):
        pass

    def __call__(self, Q, K):
        '''
        dim_Q must be same to dim_K
        :param Q: 3-d Tensor
            size([batch_size, dim_Q, T_Q])
        :param K: 3-d Tensor
            size([batch_size, dim_K, T_K])
        :return:
        '''
        assert (Q.size()[0] == K.size()[0])
        assert (Q.size()[1] == K.size()[1])
        return torch.bmm(Q.transpose(1, 2), K)

class Scale:
    def __init__(self, scale_scalar):
        self.scalar = scale_scalar

    def __call__(self, x):
        return x / self.scalar

class Mask:
    def __init__(self):
        pass

    def __call__(self, x):
        return x
