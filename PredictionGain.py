"""
@File   :   PredictionGain.py
@Date   :   10-8-202615:10
@License: See license file in the root of the repository
@Desc   : The computation of the prediction gain for a collection
of patterns and a dataset.

Copyright (c) Aki Härmä, DACS, Maastricht University, 2026.
"""
import os
import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.functional import F
from pathlib import Path

from MaxSegmentFinder import MaxSegmentFinder
from SpatialPatCorr import SpatialPatCorr
from ChannelStackers import stack_to_channels_melcc, stack_to_channels_stft

# Optimization for Blackwell
torch.set_float32_matmul_precision('medium')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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


"""
    Callable function for extracted patterns and data
"""
def measure_goodness(config):
    stacker = None
    if config["spectrum_processing"] == "melcc":
        stacker = stack_to_channels_melcc
    if config["spectrum_processing"] == "stft":
        stacker = stack_to_channels_stft

    pattern_path = f"{config["output_path"]}/clustered_patterns_{config["spectrum_processing"]}_{config["pattern_selection"]}"
    data_path = f"{config["output_path"]}/specData_{config["spectrum_processing"]}"
    outpath = f"{config["output_path"]}/goodnesses"
    os.makedirs(outpath, exist_ok=True)

    # Filenames
    data_files = sorted([str(x) for x in Path(data_path).rglob(f"*_{config["aviary"]}_*")])
    # This should be only one file
    patterns_file = sorted([str(x) for x in Path(pattern_path).rglob(f"{config["aviary"]}_aviary_patterns.pt")])[0]

    """
        Test goodness of the aviary pattern set separately for all data  
    """
    # Load patterns
    patterns = torch.load(patterns_file, weights_only=False, map_location=torch.device("cpu"))
    pkeys = list(patterns.keys())

    number_of_patterns = len(pkeys)

    df = pd.DataFrame(columns=["aviary", "data_item", "residual_energy", "original_energy"])
    cnt = 0
    for f in data_files:
        print(f"Measuring the prediction gain in {f}")
        dd = torch.load(f, weights_only=False)
        keys = [k for k in dd.keys() if dd[k]["meta"]["MIT_AST_label"] != "Speech"]
        N = len(keys)

        x = torch.stack([stacker(dd[k]) for k in keys])
        SPC = SpatialPatCorr(x.shape).to(device)

        pcorr = torch.zeros([len(pkeys), x.shape[0], x.shape[-1]])
        c0 = 0
        for p in pkeys:
            pat = patterns[p]["pat"]
            corr = SPC(x.to(device), pat.to(device))
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
            df.loc[cnt, "residual_energy"] = res.norm()
            df.loc[cnt, "original_energy"] = x.norm()
            df.loc[cnt, "data_item"] = keys[c0]
            df.loc[cnt, "aviary"] = config["aviary"]
            cnt += 1
    df.to_excel(f"{outpath}/{config["aviary"]}_{config["spectrum_processing"]}_{number_of_patterns}.xlsx", index=False)


if __name__ == '__main__':
    config = {"data_folder": "./data", "output_path": "icassp27_results", "do_spectrum_processing": False,
              "do_pattern_extraction": False, "do_pattern_selection": False, "do_goodness": True,
              "spectrum_processing": "melcc", "pattern_extraction": "extract", "pattern_selection": "tsne_kmeans",
              "goodness": "prediction_gain", "aviary": 'fl_gaia_zoo_savannah_aug2025'}

