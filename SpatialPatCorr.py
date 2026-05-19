"""
@File  : SpatialPatCorr.py
@Date  : 5/13/20261:20 PM
@License: See license file is in the root of the repository.
@Desc  :

Copyright (c) Aki Härmä, DACS, Maastricht University, 2026.
"""
import torch
from torch import nn
from torch.functional import F

def stack_to_channels(item):
    spec = torch.stack([item["spec"]-item["spec"].mean(), item["coh"]-item["coh"].mean(),
                        item["angle"]-item["angle"].mean()])
    return spec

class SpatialPatCorr(nn.Module):
    def __init__(self, dims):
        super(SpatialPatCorr, self).__init__()
        self.dims = dims
        self.eps = 0.000000001

    def single_channel(self, x, pat, unit):
        patE = pat.pow(2).sum().sqrt()
        pattern = F.conv2d(x, pat)
        norm = F.conv2d(x.pow(2), unit).sqrt()
        corr = pattern / (norm * patE + self.eps)
        padding = x.shape[-1] - corr.shape[-1] - 1
        return F.pad(corr, (1, padding))

    def forward(self, x, pat):
        unit = torch.ones([1, 1, self.dims[2], pat.shape[3]])
        corrs = torch.zeros([x.shape[0], self.dims[1], 1, self.dims[3]])
        for c1 in range(self.dims[1]):
            corrs[:,c1:c1+1,:,:] = self.single_channel(x[:,c1:c1+1,:,:], pat[:,c1:c1+1,:,:], unit)
        return corrs


if __name__ == '__main__':
    dd = torch.load("specData/spec_fl_zoo_parc_aug25_data_0.pt", weights_only=False)
    kk = list(dd.keys())
    x = stack_to_channels(dd[kk[7]]).unsqueeze(0)
    pat = x[0:1,:,:,20:64];
    SPC = SpatialPatCorr(x.shape)
    corr = SPC(x, pat)
    pcorr = corr.prod(dim=0)

