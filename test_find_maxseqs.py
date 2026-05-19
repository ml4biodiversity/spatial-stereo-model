"""
@File  : test_find_maxseqs.py
@Date  : 5/18/202611:06 AM
@License: See license file is in the root of the repository.
@Desc  :

Copyright (c) Aki Härmä, DACS, Maastricht University, 2026.
"""
import torch
from SpatialPatCorr import SpatialPatCorr, stack_to_channels
import numpy as np
import pandas as pd

"""
    Weight masks
"""
class WeightMasks():
    masks = None
    wpat = None
    number_masks = None
    def __init__(self, L, B):
        self.number_masks = int(L/B)*2-1
        self.B = B
        self.masks = np.zeros([2, self.number_masks]).astype(int)
        B05 = int(B/2)
        self.wpat = torch.zeros([self.number_masks, 3, 128, B])

        for i in range(self.number_masks):
            st = int(i*B05)
            ed = int(i*B05+B)
            self.masks[0,i] = st
            self.masks[1,i] = ed

    def mask_pat(self, pat):
        for c1 in range(self.number_masks):
            self.wpat[c1,:,:,:] = pat[0, :,:, self.masks[0,c1]:self.masks[1,c1]]
        return self.wpat

    def extract(self, pat, i):
        return pat[:,:,:,self.masks[0,i]:self.masks[1,i]]

dd = torch.load("specData/spec_fl_zoo_parc_aug25_data_0.pt", weights_only=False)
kk = list(dd.keys())
N = len(kk)

SPC = SpatialPatCorr(torch.Size([1, 3, 128, 258]))
WM = WeightMasks(258,16)

scores = torch.zeros([N, WM.number_masks])

X = torch.cat([stack_to_channels(dd[kk[c1]]).unsqueeze(0) for c1 in range(N)])

for c1 in range(N):
    pat = stack_to_channels(dd[kk[c1]]).unsqueeze(0)
    mpats = WM.mask_pat(pat)
    for c3 in range(mpats.shape[0]):
        corr = SPC(X, mpats[c3:c3 + 1,:, :, :])
        pcorr = corr.prod(dim=1)
        v = pcorr.max(dim=2).values; v[c1]=0
        scores[c1,c3] = v.max()

# Select top T patterns
T = 10
s1 = scores.max(dim=1)
df = pd.DataFrame(s1).T.astype(float).rename(columns={0:'score',1:'position'})
df = df.sort_values(by="score",ascending=False).head(T).copy()
selected = df.T.to_dict()
for k in list(selected.keys()):
    selected[k]["pat"] = WM.extract(X[int(k):int(k)+1,:,:,:],int(selected[k]["position"]))


if __name__ == '__main__':
    pass
