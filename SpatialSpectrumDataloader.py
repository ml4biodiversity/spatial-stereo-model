# -*- coding: utf-8 -*-
"""
Created on Mon Dec 30 10:45:06 2024

See LICENSE file in the root of the repository. 

Copyright (c) Aki Härmä, DACS/FSE, Maastricht University, 2024
"""

import torch
import numpy as np
from pathlib import Path
import datetime as dt

META = ['precipRate', 'pressureMax', 'dewptAvg', 'windgustHigh',
                'windspeedAvg', 'tempAve', 'humidityAvg', 'winddirAvg', 'uvHigh',
                'solarRadiationHigh', 'day_x', 'day_y']

"""
    Normalizer for meta data
"""
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
            torch.save(normalizer, f"normalizer_{str(dt.datetime.today().date())}.pt")

    def normalize(self, data):
        return torch.multiply(torch.sub(data, self.offsets), self.gains).float()

"""
    Normalizer for spectrum data (to [0,1])
"""
class SpectrumNormalizer:
    arrays = ["spec", "coh", "angle"]
    number_of_spectra = len(arrays)
    mins = np.zeros(number_of_spectra)
    maxs = np.zeros(number_of_spectra)

    def __init__(self, coefficients):
        if type(coefficients) == dict:
            self.offsets = coefficients["offsets"]
            self.gains = coefficients["gains"]
        else:
            self.mins = np.ones(self.number_of_spectra)*1000
            self.maxs = -np.ones(self.number_of_spectra)*1000
            for f in coefficients:
                dataset = torch.load(f, weights_only=False)
                for k in dataset:
                    lmins = torch.tensor([dataset[k][x].min() for x in self.arrays])
                    lmaxs = torch.tensor([dataset[k][x].max() for x in self.arrays])
                    for c1 in range(self.number_of_spectra):
                        if lmins[c1]<self.mins[c1]:
                            self.mins[c1] = lmins[c1]
                        if lmaxs[c1]>self.maxs[c1]:
                            self.maxs[c1] = lmaxs[c1]

            self.offsets = self.mins
            self.gains = 1/(self.maxs-self.mins)
            normalizer = {"offsets": self.offsets, "gains": self.gains,
                                                  "arrays": self.arrays}
            torch.save(normalizer,f"spectrum_normalizer_{str(dt.datetime.today().date())}.pt")

    def normalize(self, data):
        ndata = data.clone()
        for c1 in range(self.number_of_spectra):
            ndata[c1,:,:] = (ndata[c1,:,:]-self.offsets[c1])*self.gains[c1]
        return ndata


# The data loader
class SpatialDataset(torch.utils.data.Dataset):
  packet_index = 0
  metanorm = MetaDataNormalizer(torch.load("normalizer_2026-05-13.pt",
                                           weights_only=False))  # For metadata
  specnorm = SpectrumNormalizer(torch.load("spectrum_normalizer_2026-05-13.pt",
                                           weights_only=False))

  def __init__(self, allfiles, batch_size):
      self.keys = None
      self.N = None
      self.batch = None
      self.files = allfiles
      self.number_files = len(allfiles)
      self.file_index = 0
      self.batch_size = batch_size
      indata = torch.load(self.files[0], weights_only=False)
      self.batchify(indata)

  def stack_to_channels(self, item, st=10, ed=42):
      st = np.random.randint(90)
      ed = st + 32
      spec = torch.stack([item["spec"][:, st:ed], item["coh"][:, st:ed], item["angle"][:, st:ed]])
      return self.specnorm.normalize(spec)
      
  def batchify(self, indata):          
      self.keys = np.random.permutation(list(indata.keys()))
      self.N = len(self.keys)
      self.batch = []
      try:
        for c1 in range(int(self.N/self.batch_size)):
            self.batch.append(self.make_batch(c1, indata))
      except:
          print(f"Batchify failed at {c1} / {self.N}")
      self.N = len(self.batch)

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
      spec = torch.stack([self.stack_to_channels(data[k])
                          for k in self.keys[st:ed]])
      meta = torch.stack([self.stack_meta(data[k]["meta"])
                          for k in self.keys[st:ed]])
      meta = torch.nan_to_num(meta, 0.0)
      meta = self.metanorm.normalize(meta)

      return spec, meta
  
  def __getitem__(self, index):
      if index>=self.N-1: # When in the last packet, load new data
          self.file_index += 1
          if self.file_index == self.number_files:
              self.file_index = 0
          try:
              indata = torch.load(self.files[self.file_index], weights_only=False)
              print(f"Loaded: {self.file_index}:{self.files[self.file_index]} \n")
              self.batchify(indata)
          except:
              print(f"Batch {self.file_index} FAILED - ignoring data")
              print(f"at {self.N}/{index}")
              pass
      return self.batch[index]
  
"""
    Calling the main updates the data normalizers
"""
if __name__ == '__main__':
    print("Recomputing data normalizers...")
    files = list(Path("./specData/").glob("*.pt"))
    MetaDataNormalizer(files)
    SpectrumNormalizer(files)

   
