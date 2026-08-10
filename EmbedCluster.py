"""
@File   :   EmbedCluster.py
@Date   :   7-8-202611:19
@License: See license fuile in the root of the repository
@Desc   :
This implements
    3. Embed the pattern data into a joint vector space.
    4. Cluster candidates to $>C$ clusters.
of the ICASSP'27 Greedy Pattern Selection method

Copyright (c) Aki Härmä, DACS, Maastricht University, 2026.
"""

import os
import torch
from torch.functional import F
import numpy as np
import pandas as pd
from pathlib import Path
import torch
from sklearn.manifold import MDS, TSNE
from sklearn.cluster import HDBSCAN, KMeans
from ChannelStackers import *
from SpatialPatCorr import SpatialPatCorr


# Optimization for Blackwell
torch.set_float32_matmul_precision('medium')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def compute_distance_matrix(pats):
    stacker = pat_to_max_size
    pkeys = list(pats.keys())
    maxlen = int(np.max([pats[c0]["pat"].shape[3] for c0 in pkeys]))
    Np = len(pats)

    pcc = torch.zeros([Np, Np])
    x = torch.stack([stacker(pats[k], maxlen) for k in pkeys]).to(device)
    SPC = SpatialPatCorr(x.shape).to(device)

    for c1 in range(Np):
        d1 = x[c1:c1+1:,:,:]
        corr = SPC(x, d1)
        pcc[:,c1] = corr.max(1)[0].max(2)[0][:,0]

    pcc = torch.maximum(pcc, pcc.T)
    lpcc = (pcc + 10e-6 ).log()
    spcc = (lpcc.max() - lpcc).fill_diagonal_(0)
    return spcc


def select_patterns(patterns, D):
    # Embedding based on the distance matrix
    tsne = TSNE(n_components=2, metric="precomputed", init="random",
                perplexity=12)
    X_transformed = tsne.fit_transform(D)

    # Clustering in the embedded space
    # hdb = HDBSCAN(min_cluster_size=4)
    # xx = hdb.fit_predict(X_transformed)
    number_of_patterns = 256
    cluster = KMeans(n_clusters=number_of_patterns, random_state=0).fit(X_transformed)
    keys = patterns.keys()
    selected_patterns = {}

    for c0 in range(number_of_patterns):
        sel = [int(c) for c in np.where(cluster.labels_ == c0)[0]]
        winner = int(np.argmax([patterns[c1]["max"] for c1 in sel]))
        selected_patterns[c0] = patterns[sel[winner]]
    return selected_patterns


if __name__ == '__main__':
    inpath = "extracted_patterns"
    outpath = "clustered_patterns"
    os.makedirs(outpath, exist_ok=True)

    files = [str(f).replace("\\","/") for f in Path(inpath).glob("*.pt")]

    for f in files:
        print(f"Processing clustering and selection of patterns from {f}")
        site, day = f.replace(".pt", "").split("/")[-1].split("_day_")
        patterns = torch.load(f, weights_only=False, map_location=torch.device('cpu'))
        D = compute_distance_matrix(patterns)
        optimized_patterns = select_patterns(patterns, D)
        torch.save(optimized_patterns, outpath + "/" + site + f"_patterns_day_{day}.pt")

