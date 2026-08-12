"""
@File   :   PredictionGain.py
@Date   :   10-8-202615:10
@License: See license file in the root of the repository
@Desc   : The computation of the prediction gain for a collection
of patterns and a dataset.

Copyright (c) Aki Härmä, DACS, Maastricht University, 2026.
"""
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


pattern_path = "clustered_patterns"
data_path = "specPure"

# Filenames
data_files = sorted([str(x) for x in Path(data_path).rglob("*.pt")])
pattern_files = sorted([str(x) for x in Path(pattern_path).rglob("*.pt")])

# Load data samples
dd = torch.load(data_files[0], weights_only=False)
keys = [k for k in dd.keys() if dd[k]["meta"]["MIT_AST_label"] != "Speech"]
N = len(keys)
stacker = stack_to_channels_pure
x = torch.stack([stacker(dd[k]) for k in keys])

# Load patterns
patterns = torch.load(pattern_files[0], weights_only=False, map_location=torch.device('cpu'))
pkeys = list(patterns.keys())
pat = patterns[pkeys[0]]["pat"]


SPC = SpatialPatCorr(x.shape).to(device)

corr = SPC(x, pat)
pcorr = corr.prod(dim=1)
pos = pcorr.argmax(2)-1
el = SPC.compute_energy_loss(x.to(device), pat.to(device), pos)


if __name__ == '__main__':
    print('Hello')
