"""
@File  : DenseModel.py
@Date  : 4/20/20263:08 PM
@License: See license file is in the root of the repository.
@Desc  :

Copyright (c) Aki Härmä, DACS, Maastricht University, 2026.
"""
from torch import nn

class DenseModelEncoder(nn.Module):
    def __init__(self, input_dim, embedding_size):
        super(DenseModelEncoder, self).__init__()
        internal = 1024
        self.ffn1 = nn.Linear(input_dim.numel(), embedding_size)
        self.ffn2 = nn.Linear(embedding_size, embedding_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.ffn1(x.flatten(1,3)))
        x = self.relu(x + self.ffn2(x))
        return x


class DenseModelDecoder(nn.Module):
    def __init__(self, embedding_size, output_dim):
        super(DenseModelDecoder, self).__init__()
 
        self.outsize = [-1]+list(output_dim)
        self.ffn1 = nn.Linear(embedding_size, embedding_size)
        self.ffn2 = nn.Linear(embedding_size, output_dim.numel())
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.bn  = nn.BatchNorm1d(num_features=embedding_size)

    def forward(self, x):
        x = self.bn(x)
        x = self.relu(self.ffn1(x))
        x = self.bn(x)
        x =  self.ffn2(x)
        rx = x.view(self.outsize)
        return rx

if __name__ == '__main__':
    pass
