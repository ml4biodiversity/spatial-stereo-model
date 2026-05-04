# -*- coding: utf-8 -*-
"""
Created on Mon Dec 30 10:45:06 2024

See LICENSE file in the root of the repository. 

Copyright (c) Aki Härmä, DACS/FSE, Maastricht University, 2024
"""

import torch
import numpy as np

META = ['precipRate', 'pressureMax', 'dewptAvg', 'windgustHigh',
                'windspeedAvg', 'tempAve', 'humidityAvg', 'winddirAvg', 'uvHigh',
                'solarRadiationHigh', 'day_x', 'day_y']


class MetaDataNormalizer:
    offsets = np.zeros(len(META))
    gains = np.zeros(len(META))
    meta_labels = META

    def __init__(self, coefficients):
        table = torch.zeros([1, len(META)])
        if type(coefficients) == dict:
            self.offsets = coefficients["offsets"]
            self.gains = coefficients["gains"]
        else:
            for f in coefficients:
                dataset = torch.load(f, weights_only=False)
                for k in dataset:
                     m = torch.tensor([dataset[k]["meta"][x] for x in META])
                     table = torch.cat([table, m.unsqueeze(0)])


            self.offsets = table.mean(dim=0)
            self.gains = 1 / (table.std(dim=0) + 0.0001)
            normalizer = {"offsets": self.offsets, "gains": self.gains,
                                                  "meta": META}
            torch.save(normalizer, "normalizer_20260503.pt")

    def normalize(self, data):
        return torch.multiply(torch.sub(data, self.offsets), self.gains).float()

def stack_to_channels(item, st=10, ed=42):
    return torch.stack([item["spec"][:,st:ed], item["coh"][:,st:ed], item["angle"][:,st:ed]])

# The data loader
class SpatialDataset(torch.utils.data.Dataset):
  packet_size = 10  # the number of data files in memory
  packet_index = 0
  metanorm = MetaDataNormalizer(torch.load("normalizer_20260503.pt"))  # For metadata

  def __init__(self, allfiles, batch_size, packets=False):
      self.files = allfiles
      self.batch_size = batch_size
      if not packets:
          indata = self.load_files(allfiles)
      else:
          indata = self.load_files(allfiles[:self.packet_size])
      self.batchify(indata)
          
      
  def batchify(self, indata):          
      self.keys = np.random.permutation(list(indata.keys()))
      self.index = 0 
      self.N = int(len(self.keys)/self.batch_size)
      self.batch = []
      for c1 in range(self.N):
          self.batch.append(self.make_batch(c1, indata))          


  def __len__(self):
        'Denotes the total number of items'
        return self.N

  def input_shape(self):
      return self.batch[0][0].shape
    
  def stack_meta(self, meta):
      return torch.tensor([meta[x] for x in META])

  def make_batch(self, index, data):
      st = index*self.batch_size
      ed = st + self.batch_size
      spec = torch.stack([stack_to_channels(data[k])
                          for k in self.keys[st:ed]])
      meta = torch.stack([self.stack_meta(data[k]["meta"])
                          for k in self.keys[st:ed]])
      meta = torch.nan_to_num(meta, 0.0)
      meta = self.metanorm.normalize(meta)

      return spec, meta
  
  def load_files(self, fnames):
      dataset = torch.load(fnames[0], weights_only=False)
      for c1 in range(1,len(fnames)):
          dataset = dataset | torch.load(fnames[c1], weights_only=False)
      return dataset


  def __getitem__(self, index):
      if index==self.N-1: # When in the last packet, load new data
          self.packet_index += 1
          try:
              files = self.files[self.packet_size*self.packet_index: 
                             self.packet_size*(self.packet_index+1)]
          except: 
              files = self.files[:self.packet_size]
              self.packet_index = 0
              
          indata = self.load_files(files)
          print(f"BATCH INDEX: {self.packet_index} \n")   
          self.batchify(indata)
      
      return self.batch[index]
  
    


   
