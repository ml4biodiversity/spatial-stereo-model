"""
@File  : SpatialPatCorr.py
@Date  : 5/13/20261:20 PM
@License: See license file is in the root of the repository.
@Desc  :

Copyright (c) Aki Härmä, DACS, Maastricht University, 2026.
"""
import os
import numpy as np
import torch
from torch import nn
from torch.functional import F
from pathlib import Path
import pandas as pd
from SliceAviariesDays import *

from MaxSegmentFinder import MaxSegmentFinder

SPECMODEL = 2

# Optimization for Blackwell
torch.set_float32_matmul_precision('medium')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def stack_to_channels_melcc(item):
    spec = item["left"]
    coh = item["right"]
    angle = item["cc"]
    spec = torch.stack([spec, coh, angle])
    return spec

def stack_to_channels_stft(item):
    left = item["left"].flatten(0, 1)
    right = item["right"].flatten(0, 1)
    spec = torch.stack([left, right])
    return spec


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
            corrs[:,c1:c1+1,:,:] = self.single_channel(x[:,c1:c1+1,:,:],
                                                       pat[:,c1:c1+1,:,:], unit.to(device))
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
                try:
                    block = x[c2:c2+1, c1:c1 + 1, :, pos[c2]:pos[c2]+pat.shape[-1]]
                    ex = torch.mul(block, pat[:, c1:c1 + 1, :, :]).sum()
                    residual = block - (ex/ep)*pat[:, c1:c1 + 1, :, :]
                    energy_loss += (energy[c2:c2+1, c1:c1 + 1, pos[c2]:pos[c2]+pat.shape[-1]].sum()
                                   - residual.pow(2).sum())
                except:
                    pass
        return energy_loss


    def compute_residual_in_place(self, x, pat, pos):
        energy = x.pow(2).sum(2)
        energy_loss = 0.0
        for c1 in range(self.dims[1]):
            ep = torch.mul(pat[:, c1:c1 + 1, :, :], pat[:, c1:c1 + 1, :, :]).sum()
            for c2 in range(self.dims[0]):
                #               try:
                block = x[c2:c2 + 1, c1:c1 + 1, :, pos[c2]:pos[c2] + pat.shape[-1]]
                ex = torch.mul(block, pat[:, c1:c1 + 1, :, :]).sum()
                residual = block - (ex / ep) * pat[:, c1:c1 + 1, :, :]
                x[c2:c2 + 1, c1:c1 + 1, :, pos[c2]:pos[c2] + pat.shape[-1]] = residual
        #                except:
        #                    pass
        return x


def extract_patterns(config):
    if config["spectrum_processing"] == "melcc":
        stacker = stack_to_channels_melcc
    if config["spectrum_processing"] == "stft":
        stacker = stack_to_channels_stft
    MSF = MaxSegmentFinder()

    specPath = f"./specData_{config["spectrum_processing"]}"
    outpath = f"extracted_patterns_{config["spectrum_processing"]}"
    os.makedirs(outpath, exist_ok=True)
    data_name = config["aviary"]
    files = sorted([str(x).replace("\\","/") for x in Path(specPath).rglob(f"*_{data_name}_*")])
    for f in files:
        print(f"Processing {f}")
        dd = torch.load(f, weights_only=False)
        kk = list(dd.keys())
        N = len(kk)
        x = torch.stack([stacker(dd[kk[c1]]) for c1 in range(N)
                         if dd[kk[c1]]["meta"]["MIT_AST_label"] != "Speech"]).to(device)
        N = x.shape[0]
        energy = x.pow(2).sum(2)
        patterns = {}

        SPC = SpatialPatCorr(x.shape).to(device)

        for c2 in range(N):
            try:
                s, pat0 = MSF.process((x[c2, 0, :, :].detach().cpu().abs()+0.000001).log(), maxseglen=32)
                xpat = subtract_mean(x[c2:c2 + 1, :, :, s[0]:s[1]])
                corr = SPC(x, xpat)
                pcorr = corr.prod(dim=1)
                pos = pcorr.argmax(2) - 1
                el = SPC.compute_energy_loss(x.to(device), xpat.to(device), pos)
                patterns[c2] = {"pat": x[c2, :, :, s[0]:s[1]].unsqueeze(0),
                                "pos": s[0], "max": el}
            except:
                print(f"Something broken in {f}  file {c2}/{N} - omitting")
                break

        torch.save(patterns, f"{outpath}/sel_pat_{f.split("/")[1][:-3]}.pt")

if __name__ == '__main__':
    dpath = f"specData{SPECMODEL}"
    outpath = f"extracted_patterns_{SPECMODEL}"
    os.makedirs(outpath, exist_ok=True)
    aviaries = pd.read_excel("ICASSP27_birds.xlsx",index_col=0)
    aviaries = aviaries["preprocessed_new"].unique()
    stacker = stack_to_channels
    MSF = MaxSegmentFinder()

    for avi in aviaries:
        files = sorted([str(x).replace("\\\\","/") for x in Path(dpath).rglob(f"*_{avi}_*")])
        print(f"Processing set {avi} of {len(files)} files")
        for f in files:
            print(f"Processing {f}")
            dd = torch.load(f, weights_only=False)
            kk = list(dd.keys())
            N = len(kk)
            x = torch.stack([stacker(dd[kk[c1]]) for c1 in range(N)
                             if dd[kk[c1]]["meta"]["MIT_AST_label"] != "Speech"]).to(device)
            N = x.shape[0]
            energy = x.pow(2).sum(2)
            patterns = {}

            SPC = SpatialPatCorr(x.shape).to(device)

            for c2 in range(N):
                try:
                    s, pat0 = MSF.process(x[c2,0,:,:].detach().cpu().abs().log(), maxseglen=32)
                    xpat = subtract_mean(x[c2:c2+1,:,:,s[0]:s[1]])
                    corr = SPC(x, xpat)
                    pcorr = corr.prod(dim=1)
                    pos = pcorr.argmax(2)-1
                    el = SPC.compute_energy_loss(x.to(device), xpat.to(device), pos)
                    patterns[c2] = {"pat":x[c2, :, :, s[0]:s[1]].unsqueeze(0),
                                    "pos":s[0], "max":el}
                except:
                    print(f"Something broken in {f}  file {c2}/{N} - omitting")
                    break

            torch.save(patterns,f"{outpath}/sel_pat_{f.split("/")[1][:-3]}.pt")
