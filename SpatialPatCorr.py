"""
@File  : SpatialPatCorr.py
@Date  : 5/13/20261:20 PM
@License: See license file is in the root of the repository.
@Desc  :

This implements
     1. Select >>C pattern candidates.
     2. Score the goodness of the patterns on data.

of the ICASSP'27 Greedy Pattern Selection method

Note: This script does not yet perform scoring at a day level - we do that now
in steps 3-4!

Copyright (c) Aki Härmä, DACS, Maastricht University, 2026.
"""
import numpy as np
import torch
from torch import nn
from torch.functional import F
from pathlib import Path

from MaxSegmentFinder import MaxSegmentFinder
from ChannelStackers import *

# Optimization for Blackwell
torch.set_float32_matmul_precision('medium')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def subtract_mean(x):
    return torch.subtract(x[:, :, :, :].permute([0, 3, 2, 1]),
                          x.mean([2, 3])[0]).permute([0, 3, 2, 1])


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
        unit = torch.ones([1, 1, self.dims[2], pat.shape[3]]).to(device)
        corrs = torch.zeros([x.shape[0], self.dims[1], 1, self.dims[3]])
        for c1 in range(self.dims[1]):
            corrs[:,c1:c1+1,:,:] = self.single_channel(x[:,c1:c1+1,:,:], pat[:,c1:c1+1,:,:], unit)
        return corrs

    def compute_gain(self, x, pat):
        gains = torch.zeros([x.shape[0], self.dims[1], 1, self.dims[3]])
        for c1 in range(self.dims[1]):
            g = torch.mul(pat[:,c1:c1+1,:,:], pat[:,c1:c1+1,:,:]).sum()
            gg = F.conv2d(x[:,c1:c1+1,:,:], pat[:,c1:c1+1,:,:])/g
            padding = x.shape[-1] - gg.shape[-1] - 1
            gains[:, c1:c1 + 1, :, :] = F.pad(gg, (1, padding))
        return gains.mean(1)

    def compute_energy_loss(self, x, pat, pos):
        energy = x.pow(2).sum(2)
        energy_loss = 0.0
        for c1 in range(self.dims[1]):
            ep = torch.mul(pat[:, c1:c1 + 1, :, :], pat[:, c1:c1 + 1, :, :]).sum()
            for c2 in range(self.dims[0]):
 #               try:
                block = x[c2:c2+1, c1:c1 + 1, :, pos[c2]:pos[c2]+pat.shape[-1]]
                ex = torch.mul(block, pat[:, c1:c1 + 1, :, :]).sum()
                residual = block - (ex/ep)*pat[:, c1:c1 + 1, :, :]
                energy_loss += (energy[c2:c2+1, c1:c1 + 1, pos[c2]:pos[c2]+pat.shape[-1]].sum()
                               - residual.pow(2).sum())
#                except:
#                    pass
        return energy_loss

if __name__ == '__main__':
    dpath = "specPure/"
    files = sorted([str(x) for x in Path(dpath).rglob("*.pt")])
    B = 4
    Nb = int(len(files)/B)

    for c0 in range(Nb):
        print(f"Processing set {c0}/Nb")
        dd = torch.load(files[B*c0], weights_only=False)
        for c1 in range(B*c0+1, B*c0+B):
            dd = dd|torch.load(files[c1], weights_only=False)

        keys = [k for k in dd.keys() if dd[k]["meta"]["MIT_AST_label"] != "Speech"]
        N = len(keys)
        stacker = stack_to_channels_pure
        x = torch.stack([stacker(dd[k]) for k in keys])

        energy = x.pow(2).sum(2)
        patterns = {}
        MSF = MaxSegmentFinder()
        SPC = SpatialPatCorr(x.shape).to(device)

        for c1 in range(N):
            print(f"Processing {c1}/{N}")
            try:
                s, pat0 = MSF.process(x[c1,0,:,:], maxseglen=32)
                xpat = subtract_mean(x[c1:c1+1,:,:,s[0]:s[1]])
                corr = SPC(x, xpat)
                pcorr = corr.prod(dim=1)
                pos = pcorr.argmax(2)-1
                el = SPC.compute_energy_loss(x.to(device), xpat.to(device), pos)
                patterns[c1] = {"pat":x[c1, :, :, s[0]:s[1]].unsqueeze(0), "pos":s[0], "max":el,
                                "key":keys[c1], "meta":dd[keys[c1]]["meta"]}
            except:
                print(f"Something broken in {c1}/{N} - omitting")
                break
        torch.save(patterns,f"selected_patterns_{c0}.pt")
