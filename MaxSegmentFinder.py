# -*- coding: utf-8 -*-
"""
Created on Fri Sep  5 09:35:12 2025

See LICENSE file in the root of the repository. 

Copyright (c) Aki Härmä, DACS/FSE, Maastricht University, 2024
"""

import torch
import numpy as np
import matplotlib.pyplot as plt

class MaxSegmentFinder():
    def find_max_segment(self, sig):
        dsig = np.r_[1, np.diff(1*(sig>0))]
        dsig[-1] = -1
        up = np.where(dsig==1)[0]
        down = np.where(dsig==-1)[0]
        N = min([len(up), len(down)])
        energies = [sig[up[c1]:down[c1]].sum() for c1 in range(N)]
        maxseg = np.argmax(energies)
        return up[maxseg], down[maxseg]

    def process(self, spec, maxseglen=20):
        env = spec.abs().sum(axis=0)
        env = env-env.min()
        maxenv = env.max()
        found = False
        for th in np.arange(maxenv):
            up, down = self.find_max_segment(env-th)            
            if down-up<maxseglen:
                found = True
                break
        if not found: 
            return None, None            
        return [up, down], spec[:,up:down]
            

"""
    Test
"""
if __name__ == '__main__':
    file = "specData/spec_fl_zoo_eindhoven_20250308_metadataspeechless_with_perch_0.pt"
    ss = torch.load(file,weights_only=False)
    keys = list(ss.keys())
    
    k = 241
    spec = ss[keys[k]]["spec"]
    
    MSF = MaxSegmentFinder()
    s, pat = MSF.process(spec, maxseglen=32)



