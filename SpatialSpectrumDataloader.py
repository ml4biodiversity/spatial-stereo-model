# -*- coding: utf-8 -*-
"""
Created on Mon Dec 30 10:45:06 2024

See LICENSE file in the root of the repository. 

Copyright (c) Aki Härmä, DACS/FSE, Maastricht University, 2024
"""

import torch
import numpy as np


def stack_to_channels(item):
    return torch.stack([item["spec"], item["coh"], item["angle"]])

# The data loader
class SpatialDataset(torch.utils.data.Dataset):
  'Characterizes a dataset for PyTorch' 
  def __init__(self, indata, batch_size):
      self.data = indata
      self.keys = np.random.permutation(list(indata.keys()))
      self.index = 0 
      self.N = int(len(self.keys)/batch_size)
      self.batch_size = batch_size
      self.batch = []
      for c1 in range(self.N):
          self.batch.append(self.make_batch(c1))

  def __len__(self):
        'Denotes the total number of items'
        return self.N

  def input_shape(self):
      return self.batch[0][0].shape
    
  def stack_meta(self, meta):
      return torch.tensor([meta[x] for x in ["tempAve",
                                             "solarRadiationHigh","precipRate"]])

  def make_batch(self, index):
      st = index*self.batch_size
      ed = st + self.batch_size
      spec = torch.stack([stack_to_channels(self.data[k])
                          for k in self.keys[st:ed]])
      meta = torch.stack([self.stack_meta(self.data[k]["meta"])
                          for k in self.keys[st:ed]])
      meta = torch.nan_to_num(meta, 0.0)
      return spec, meta

  def __getitem__(self, index):
      return self.batch[index]
  
    


   
