"""
@File   :   ChannelStackers.py
@Date   :   7-8-202612:32
@License: See license file in the root of the repository
@Desc   : Methods for stacking spectrum data into tensors

Copyright (c) Aki Härmä, DACS, Maastricht University, 2026.
"""

import torch

def stack_to_channels_stft(item):
    left = item["left"].flatten(0,1)
    right = item["right"].flatten(0,1)
    spec = torch.stack([left, right])
    return spec

def pat_to_max_size(item, maxsize):
    d = item["pat"].shape
    spec = torch.zeros([d[1], d[2], maxsize])
    spec[:, :, :d[3]] = item["pat"][0,:,:,:]
    return spec

def stack_to_channels_mel_spatial(item):
    spec = item["spec"]
    coh = item["coh"]
    angle = item["angle"]
    # spec = torch.stack([left-left.mean(), right-right.mean()])
    spec = torch.stack([spec, coh, angle])
    return spec

def stack_to_channels_melcc(item):
    spec = item["left"]
    coh = item["right"]
    angle = item["cc"]
    spec = torch.stack([spec, coh, angle])
    return spec