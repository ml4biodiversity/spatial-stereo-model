"""
@File   :   PredictionGain.py
@Date   :   10-8-202615:10
@License: See license file in the root of the repository
@Desc   : The computation of the prediction gain for a collection
of patterns and a dataset.

Copyright (c) Aki Härmä, DACS, Maastricht University, 2026.
"""
import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.functional import F
from pathlib import Path

from MaxSegmentFinder import MaxSegmentFinder
from ChannelStackers import *
from SpatialPatCorr import SpatialPatCorr

# Optimization for Blackwell
torch.set_float32_matmul_precision('medium')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SPECMODEL = 1

def compute_residual(xin, patterns, pkeys, df):
    x = xin.clone()
    energy = x.pow(2).sum(2)
    E = []
    maxx = x.shape[-1]
    for c1 in range(df.shape[0]):
        item = df.loc[c1]
        pat = patterns[pkeys[int(item["pattern"])]]["pat"]
        st = int(item["location"])
        ed = int(item["location"] + pat.shape[-1])
        if ed > maxx:
            continue
        for c2 in range(x.shape[1]):
            ep = torch.mul(pat[:, c2:c2 + 1, :, :], pat[:, c2:c2 + 1, :, :]).sum()
            block = x[:, c2:c2 + 1, :, st:ed]
            ex = torch.mul(block, pat[:, c2:c2 + 1, :, :]).sum()
            residual = block - (ex / ep) * pat[:, c2:c2 + 1, :, :]
            x[0, c2:c2 + 1, :, st:ed] = residual
        E.append(x.norm())
    return x, E





pattern_path = f"clustered_patterns"
data_path = f"specData_{SPECMODEL}"

# Filenames
data_files = sorted([str(x) for x in Path(data_path).rglob("*.pt")])
pattern_files = sorted([str(x) for x in Path(pattern_path).rglob("*.pt")])

# Load data samples
dd = torch.load(data_files[0], weights_only=False)
for df in data_files[1:10]:
    dd = dd|torch.load(df, weights_only=False, map_location=torch.device('cpu'))

keys = [k for k in dd.keys() if dd[k]["meta"]["MIT_AST_label"] != "Speech"]
N = len(keys)
stacker = stack_to_channels_pure
x = torch.stack([stacker(dd[k]) for k in keys])

# Load patterns
patterns = torch.load(pattern_files[0], weights_only=False, map_location=torch.device('cpu'))
for pf in pattern_files[1:]:
    patterns = patterns | torch.load(pf, weights_only=False, map_location=torch.device('cpu'))
pkeys = list(patterns.keys())

CRE = []
E = []
SPC = SpatialPatCorr(x.shape).to(device)

pcorr = torch.zeros([len(pkeys), x.shape[0], x.shape[-1]])
c0 = 0
for p in pkeys:
    pat = patterns[p]["pat"]
    corr = SPC(x, pat)
    pcorr[c0, :, :] = corr.max(dim=1)[0][:,0,:]
    c0 += 1

res = x.clone()
RE = [x.norm()]
for c0 in range(x.shape[0]):
    mv = pcorr[:,c0,:].max(dim=0)
    df = pd.DataFrame(data={"pattern":[int(x) for x in mv[1]],
                            "corr":mv[0]}).sort_values(by="corr",ascending=False).reset_index(drop=False)
    df = df.rename(columns={"index":"location"})
    res[c0:c0+1,:,:,:], e = compute_residual(x[c0:c0+1,:,:,:], patterns, pkeys, df)
    RE.append(res.norm())



if __name__ == '__main__':
    print('Hello')
