"""
@File   :   MeteoSpectroStream.py
@Date   :   17-6-202611:04
@License: See license fuile in the root of the repository
@Desc   : 

Copyright (c) Aki Härmä, DACS, Maastricht University, 2026.
"""
import soundfile
import torch
from pathlib import Path
from torch import nn
from torch.distributed._shard.sharding_spec.chunk_sharding_spec_ops import embedding

from SpectroStream import SpectroStreamEncoder, SpectroStreamDecoder
import torch.nn.functional as F
from MeteoSpectrumDataloader import MeteoSpectrumDataset
from torch.nn.utils import weight_norm

def load_audio_torch(fname):
    if fname.find("er_path")>-1:
        fname = fname.replace("er_path/","")
    s, fs = soundfile.read(fname)
    return torch.FloatTensor(s.T), fs


class MeteoSpectroStream(nn.Module):
    """
    The main SpectroStream model that combines the Encoder and Decoder.
    It handles the end-to-end process from audio to audio.
    """

    def __init__(self, C0=32, D=128):
        super().__init__()
        self.encoder = SpectroStreamEncoder(C0, D)
        self.decoder = SpectroStreamDecoder(C0, D)

        # STFT parameters based on the paper's description
        # hop_length is 480 (from 100Hz frame rate at 48kHz).
        # win_length is 2x hop_length due to 2x overlapping factor.
        self.hop_length = 480
        self.win_length = 960
        self.n_fft = 960
        self.n_meta = 8
        self.meteo = nn.Linear(self.n_fft, self.n_fft)

    # Signals to spectrum representation
    def to_spectrogram(self, x):
        x_left_stft = torch.stft(x[:, 0],
                                 n_fft=self.n_fft,
                                 hop_length=self.hop_length,
                                 win_length=self.win_length,
                                 return_complex=True)
        x_right_stft = torch.stft(x[:, 1],
                                  n_fft=self.n_fft,
                                  hop_length=self.hop_length,
                                  win_length=self.win_length,
                                  return_complex=True)

        # --- STEP 2: Pre-process Spectrogram for the Network ---
        # "omit the highest (Nyquist) frequency to avoid odd number of bins"
        x_left_stft = x_left_stft[:, :
                                  -1, :]  # Shape becomes (B, 480, T_frames)
        x_right_stft = x_right_stft[:, :-1, :]

        # Convert complex spectrograms to 2-channel real tensors for the Conv2D layers
        # The shape needs to be (B, C, T, F) -> (B, 2, T_frames, Freq_bins)
        def to_channels(s):
            # stack real and imaginary parts as channels
            s_ch = torch.stack([s.real, s.imag], dim=1)
            # permute from (B, 2, F, T) to (B, 2, T, F)
            return s_ch.permute(0, 1, 3, 2)

        x_left_ch = to_channels(x_left_stft)
        x_right_ch = to_channels(x_right_stft)
        return x_left_ch, x_right_ch


    # To waveform representation
    def to_waveform(self,out_left_stft_ch, out_right_stft_ch ):
        # --- STEP 7: Post-process Spectrogram for Audio ---
        # Convert the 2-channel (real, imag) tensors back to complex numbers
        def to_complex(s_ch):
            # permute from (B, 2, T, F) back to (B, T, F, 2)
            s = s_ch.permute(0, 2, 3, 1)
            return torch.complex(s[..., 0], s[..., 1]).permute(0, 2, 1)

        out_left_stft_complex = to_complex(out_left_stft_ch)
        out_right_stft_complex = to_complex(out_right_stft_ch)

        # Add back the Nyquist frequency bin (as a row of zeros) before iSTFT
        out_left_stft = F.pad(out_left_stft_complex, (0, 0, 0, 1))
        out_right_stft = F.pad(out_right_stft_complex, (0, 0, 0, 1))

        # --- STEP 8: Spectrogram -> Audio (iSTFT) ---
        # Perform inverse STFT to get the final audio waveform
        out_left_wav = torch.istft(out_left_stft,
                                   n_fft=self.n_fft,
                                   hop_length=self.hop_length,
                                   win_length=self.win_length)
        out_right_wav = torch.istft(out_right_stft,
                                    n_fft=self.n_fft,
                                    hop_length=self.hop_length,
                                    win_length=self.win_length)

        # --- STEP 9: Final Output ---
        # Combine the left and right channels into a final stereo waveform
        out_wav = torch.stack([out_left_wav, out_right_wav], dim=1)

        return out_wav

    def forward(self, x, m):
        zl = x["left"].permute([0, 1, 3, 2])
        zr = x["right"].permute([0, 1, 3, 2])

        c = model.encoder(zl, zr)
        meta = m.unsqueeze(2).repeat(1, 1, c.shape[2])
        embedding = torch.cat([c, meta], dim=1)

        # Feedforward merging of tokens and metadata
        out = ffn(embedding.permute([0, 2, 1])).permute([0, 2, 1])

        # Decoding
        out_left_stft_ch, out_right_stft_ch = model.decoder(out[:, :, :])

        mix = (out_left_stft_ch + out_right_stft_ch)[:, 0, :, :].abs().log()
        return mix

if __name__ == '__main__':
    model = MeteoSpectroStream()
    loss = torch.nn.MSELoss()
    files = list(Path("./specData2/").glob("*.pt"))
    ffn = nn.Linear(140,128)

    loader = MeteoSpectrumDataset(files, 16)
    x, m = loader[1]
    y = x["cc"].permute([0, 2, 1])[:,1:, :].log()

    p = model(x, m)
    e = loss(y, p)
    




