"""
@File  : VITModel.py
@Date  : 5/4/202611:11 AM
@License: See license file is in the root of the repository.
@Desc  :

Copyright (c) Aki Härmä, DACS, Maastricht University, 2026.
"""

import torch
from torch import nn
from transformers import ViTConfig, ViTModel
# from SpatialSpectrumDataloader.SpatialDataset import stack_to_channels


class ViTEncoder(nn.Module):
    def __init__(self, input_dim, embedding_size):
        super(ViTEncoder, self).__init__()
        self.configuration = ViTConfig(image_size=input_dim, num_hidden_layers=12)
        self.model = ViTModel(self.configuration)
        self.ffn = nn.Linear(51*768, embedding_size)

    def forward(self, x):
        z = self.model(x)
        # y = self.ffn(z["pooler_output"])
        y = self.ffn(z["last_hidden_state"].flatten(start_dim=1))
        return y


if __name__ == '__main__':
    image_size = 128
    embedding_size = 2048
    model = ViTEncoder(image_size, embedding_size=embedding_size)
    d = torch.load("specData/spec_fl_zoo_parc_aug25_data_0.pt", weights_only=False)
    #cc = stack_to_channels(d[list(d.keys())[0]], st=10, ed=138)
    #y = model(cc.unsqueeze(0))

