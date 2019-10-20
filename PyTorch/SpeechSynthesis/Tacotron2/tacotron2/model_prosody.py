from math import sqrt
import torch
from torch.autograd import Variable
from torch import nn
from torch.nn import functional as F
from torch.nn.modules import MultiheadAttention
import sys
from os.path import abspath, dirname
# enabling modules discovery from global entrypoint
sys.path.append(abspath(dirname(__file__)+'/../'))
from common.layers_prosody import *
from common.utils import to_gpu, get_mask_from_lengths

class ScaledDotProductAttention(nn.Module):
    def __init__(self, scale_scalar):
        super(ScaledDotProductAttention, self).__init__()
        self.MatMul = MatMul()
        self.Scale = Scale(scale_scalar)
        self.Mask = Mask()
        self.mask_valid = True

    def mask_on(self):
        self.mask_valid = True

    def mask_off(self):
        self.mask_valid = False

    def forward(self, Q, K, V):
        '''

        :param Q: 3-d Tensor
            size([batch_size, dim_Q, T_Q])
        :param K: 3-d Tensor
            size([batch_size, dim_K, T_K])
        :param V: 3-d Tensor
            size([batch_size, dim_V, T_V])
        :return:
        '''
        x = self.MatMul(Q, K)
        x = self.Scale(x)
        x = self.Mask(x) if self.mask_valid == True
        x = torch.softmax(x, dim=2) # x.size([batch_size, T_Q, T_K])
        x = self.MatMul(x, V.transpose(1, 2)) # x.size([batch_size, T_Q, d_v])



class MultiheadAttention(nn.Module):
    def __init__(self, num_heads):
        super(MultiheadAttention, self).__init__()



class StyleAttention(nn.Module):
    def __init__(self,
                 mode='multihead',
                 modes_list=['content_based', 'dot_product', 'location_based', 'multihead']
                 ):
        super(StyleAttention, self).__init__()
        if mode not in modes_list:
            raise ValueError('Invalid style sttention mode selected.')
        if mode == 'multihead':
            return



class StyleTokenLayer(nn.Module):
    def __init__(self, dim_style_embedding, num_tokens,):
        super(StyleTokenLayer, self).__init__()
        # initialize GSTs with randomly initialized embeddings
        self.GlobalStyleTokens = torch.randn([dim_style_embedding, num_tokens])
        self.attention =


    def forward(self, *input):


class SpeakerEmbeddingLookup:
    def __init__(self, speaker_embeddings):
        '''
        :param speaker_embeddings: 2-d Tensor.
             Given speaker embedding. size([dim_speaker_embedding, total number of speakers])
        '''
        self.speaker_embeddings = speaker_embeddings
    def __call__(self, speaker_id):
        '''
        Look up speaker embedding corresponding to given speaker id.
        :param speaker_id: int
            # of speaker id.
        :return: 1-d Tensor
            size([dim_speaker_embedding])
        '''
        return self.speaker_embeddings[:, speaker_id]

class ReferenceEncoder(nn.Module):
    def __init__(self, dim_ref, len_ref, in_sizes=(1, 32, 32, 64, 64, 128), out_sizes=(32, 32, 64, 64, 128, 128), rnn_mode='GRU', rnn_units=128, dim_prosody=128):
        super(ReferenceEncoder, self).__init__()
        # 6-Layer Strided Conv2D w/ BatchNorm
        assert(len(in_sizes) == len(out_sizes))
        self.layers = nn.ModuleList()
        layers_per_block = 3
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
        self.rnn = nn.RNNBase(mode=rnn_mode, input_size=out_sizes[-1] * int(dim_r / 2**(len(self.layers) / layers_per_block)),hidden_size=rnn_units)
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