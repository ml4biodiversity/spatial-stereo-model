# -*- coding: utf-8 -*-
"""
Created on Wed May 21 11:35:37 2025

See LICENSE file in the root of the repository. 

Copyright (c) Aki Härmä, DACS/FSE, Maastricht University, 2024
"""

import os
import torchaudio as ta
import torch
from torch.utils import data
import numpy as np
from pathlib import Path
import datetime
import pandas as pd
import subprocess
from shutil import copyfile
import scipy.signal as dsp
import soundfile
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


FS = 24000
stereo_condition_threshold = 0.2
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_audio_torch(fname):
    #if fname.find("er_path")>-1:
    #    fname = fname.replace("er_path/","")

    s, fs = soundfile.read(fname)
    return torch.FloatTensor(s.T), fs


def get_features(infeat):
    feat = infeat.copy()
    year_time = feat["datetime"].timetuple().tm_yday
    feat["year_x"] = np.sin(2*np.pi*year_time/365)
    feat["year_y"] = np.cos(2*np.pi*year_time/365)
    
    t = feat["datetime"].timetuple()
    day_time = t.tm_hour+t.tm_min/60+t.tm_sec/(3600)/24
    feat["day_x"] = np.sin(2*np.pi*day_time)
    feat["day_y"] = np.cos(2*np.pi*day_time)
       
   # fields = ['datetime', 'precipRate', 'pressureMax', 'dewptAvg', 'windgustHigh',
   #        'windspeedAvg', 'tempAve', 'humidityAvg', 'winddirAvg', 'uvHigh',
   #        'solarRadiationHigh', 'lon', 'lat', 'MIT_AST_label', 'perch_prediction',
   #        'birdnet_prediction', 'overlap', 'fusion_model_prediction', 'year_x',
   #        'year_y','day_x','day_y']

    fields = ['datetime', 'precipRate', 'pressureMax', 'dewptAvg', 'windgustHigh',
              'windspeedAvg', 'tempAve', 'humidityAvg', 'winddirAvg', 'uvHigh',
              'solarRadiationHigh', 'lon', 'lat', 'MIT_AST_label', 'year_x',
              'year_y', 'day_x', 'day_y']


    return feat[fields]


class SpectrumProcessor(data.Dataset):
    out_path = "garden_sets" 
    B = 64
    N = 65
    Nmel = 128
    meta = None
    def __init__(self):          
            self.specProcessor = ta.transforms.Spectrogram(n_fft=2048,
                                                           hop_length=256,
                                                           power=None)                
            self.melProcessor = ta.transforms.MelSpectrogram(sample_rate=FS,
                                                           n_fft=2048,
                                                           hop_length=256,
                                                           n_mels=self.Nmel,
                                                           norm="slaney")
            self.resampler = ta.transforms.Resample(orig_freq=48000, new_freq=FS)
    
    
    def process_signal(self, sig):        
        # Resample
        sig = self.resampler(sig)
        # Pre-epmhasis
        sig = torch.FloatTensor(dsp.lfilter([1,-0.95],[1, -0.6], sig))
        # Complex spectrum                           
        f = self.specProcessor(sig)
        # Cross-correlation spectra
        cc = torch.mul(f[0,:,:].conj(),f[1,:,:])    
        sp = self.melProcessor(sig[0,:]+sig[1,:])
        # Sum real and complex parts
        rmcc = self.melProcessor.mel_scale(cc.real)
        imcc = self.melProcessor.mel_scale(cc.imag)
        # Absolute coherence weight
        z = torch.complex(rmcc, imcc).abs()
        ang = torch.complex(rmcc, imcc).angle()                            
        sp = self.melProcessor(sig[0,:]+sig[1,:])
        lsp = 20*np.log(np.abs(sp)+0.00001)
        lsp[lsp<-120]=-120
        return lsp, z, ang/np.pi
                         
        
    def __len__(self):
        'Denotes the total number of items'
        return self.N

"""
    Narrow-band full spectrum model
"""
class PureSpectrumProcessor(data.Dataset):
    Nfft = 1024
    hop = 512
    B = 400
    start_bin = 8
    end_bin = start_bin+B
    meta = None
    def __init__(self):
        self.specProcessor = ta.transforms.Spectrogram(n_fft=self.Nfft,
                                                       hop_length=self.hop,
                                                       power=None)
        self.resampler = ta.transforms.Resample(orig_freq=48000, new_freq=FS)

    def process_signal(self, sig):
        # Resample
        sig = self.resampler(sig)
        # Pre-epmhasis
        sig = torch.FloatTensor(dsp.lfilter([1,-0.95],[1, -0.6], sig))
        # Complex spectrum
        f = self.specProcessor(sig)
        cc = torch.mul(f[:, self.start_bin:self.end_bin, :].conj(),
                       f[:, self.start_bin:self.end_bin, :]).real.abs()+0.00000001
        ang = (f[1, self.start_bin:self.end_bin, :].angle()
                -f[0, self.start_bin:self.end_bin, :].angle())
        return cc[0,:,:].abs().log(), cc[1,:,:].abs().log(), ang


""" 
     The processing
"""            
def raw_file_processing(specProc, meta, data_name, data_path, spec_path):
    StoreSize = 1000

    N = meta.shape[0]
    NumStores = int(np.ceil(N/StoreSize))
    cnt = 0
    # Slice to files of StoreSize blocks
    for s1 in range(NumStores):
        print(f"Processing set {s1} of {data_name}")
        #if os.path.exists(f"{spec_path}/spec_{data_name}_{0}.pt"):
        #    print(f"Set {data_name} already done - exiting!")
        #    break
        specData = {}
        #for c1 in range(meta.shape[0]):
        for c1 in range(StoreSize):            
            if cnt == N:
                break
            file = meta.loc[cnt,"filename"]

            if file in ["file removed", "file missing"]:
                cnt += 1
                continue            
            try:
                cnt += 1
                mets = get_features(meta.loc[cnt])
                afile = f"{data_path}/{meta.loc[cnt, 'filename']}"
                sig,fs = load_audio_torch(afile)
                # If the stereo signal is broken, we skip the sample
                c = sig.norm(dim=1)
                if ((c[0] - c[1]).abs() / (c[0] + c[1])) > stereo_condition_threshold:
                    continue
                lsp, cc, ang = specProc.process_signal(sig)
                if lsp.isnan().any():
                    print(f"Failed in {afile}" )
            except:
                print(f"corrupted or removed audio file: {file}")          
                continue
            # Storing the coherence spectrogram            
            specData[file] = {"spec":lsp, "coh":cc, "angle":ang, "meta":mets}
   
        torch.save(specData, f"{spec_path}/spec_{data_name}_{s1}.pt")            
                

"""
    Main script for coherent spectrum processing
"""
if __name__ == '__main__':
    fpath = "/media/kakskyt/data/zoodata/er_path"

    spec_path1 ="specData1"
    os.makedirs(spec_path1,exist_ok=True)
    spec_path2 ="specData2"
    os.makedirs(spec_path2,exist_ok=True)

    files = [str(x) for x in Path(fpath).rglob("*_metadata.xlsx")]

    files = [x for x in files if x.find("flamingo")>-1]

    specProc1 = SpectrumProcessor()
    specProc2 = PureSpectrumProcessor()

    for f in files:
        meta = pd.read_excel(f)
        data_name = f[f.rfind("/") + 1:f.rfind("meta") - 1]
        # raw_file_processing(specProc1, meta, data_name, fpath, spec_path1)
        raw_file_processing(specProc2, meta, data_name, fpath, spec_path2)

    

