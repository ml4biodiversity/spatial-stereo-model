"""
@File  : DenseModel.py
@Date  : 4/20/20263:08 PM
@License: See license file is in the root of the repository.
@Desc  :

Copyright (c) Aki Härmä, DACS, Maastricht University, 2026.
"""
from torch import nn

"""
    Encoder based on 2-d convolution modules
"""
class Cnn2ModelEncoder(nn.Module):
    def __init__(self, input_dim, embedding_size):
        super(Cnn2ModelEncoder, self).__init__()
        in_channels = input_dim[0]
        cnn_channels1 = 64
        cnn_channels2 = 16
        cnn_channels3 = 8
        stride = 1
        fnn_input = 896
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, cnn_channels1, kernel_size=5, stride=stride, padding=1),
            nn.Dropout(0.1),
            nn.BatchNorm2d(cnn_channels1),
            nn.ReLU())
        self.maxpool = nn.MaxPool2d(kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Sequential(
            nn.Conv2d(cnn_channels1, cnn_channels2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(cnn_channels2))

        self.conv3 = nn.Sequential(
            nn.Conv2d(cnn_channels2, cnn_channels3, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(cnn_channels3))

        self.avgpool = nn.AvgPool2d(3, stride=1)
        self.ffn1 = nn.Linear(fnn_input, embedding_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.maxpool(x)
        x = self.conv3(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.ffn1(x)
        return x


class Cnn2ModelDecoder(nn.Module):
    def __init__(self, embedding_size, output_dim):
        super(Cnn2ModelDecoder, self).__init__()        
        self.outsize = [-1]+list(output_dim)
        self.ffn1 = nn.Linear(embedding_size, output_dim.numel())
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.ffn1(x))
        rx = x.view(self.outsize)
        return rx

if __name__ == '__main__':
    pass


def tmp():
    x = encoder.conv1(x)
    x = encoder.maxpool(x)
    x = encoder.conv2(x)
    x = encoder.maxpool(x)
    x = encoder.conv3(x)
    x = encoder.avgpool(x)
    x = x.view(x.size(0), -1)
    x = encoder.ffn1(x)
