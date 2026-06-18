"""
@File   :   MeteoSpectroStream.py
@Date   :   17-6-202611:04
@License: See license fuile in the root of the repository
@Desc   : 

Copyright (c) Aki Härmä, DACS, Maastricht University, 2026.
"""
import soundfile
import torch
from torch import nn
from SpectroStream import SpectroStreamEncoder, SpectroStreamDecoder
import torch.nn.functional as F
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

    def forward(self, x):
        """
        The main forward pass of the SpectroStream model.

        Args:
            x (Tensor): The input stereo audio waveform.
                        Shape: (B, 2, T_samples), where B is batch size,
                               2 is for stereo, T_samples is the number of audio samples.
        Returns:
            Tensor: The reconstructed stereo audio waveform.
                    Its length will be slightly shorter than the input due to latency.
        """
        # --- STEP 1: Audio -> Spectrogram (STFT) ---
        # Process left and right channels separately

        x_left_ch, x_right_ch = self.to_spectrogram(x)

        # --- STEP 3: Encoding ---
        # Pass the prepared spectrograms through the encoder
        embedding = self.encoder(x_left_ch, x_right_ch)

        # --- STEP 4: Quantization (Placeholder) ---
        # In a real codec, a quantizer (like RVQ) would be used here.
        # This step compresses the continuous embedding into discrete codes.
        # embedding -> codes -> quantized_embedding
        # For this autoencoder implementation, we pass it through directly.
        quantized_embedding = embedding

        # --- STEP 5: Decoder Look-ahead ---
        # "we give the decoder a one-embedding look-ahead by shifting its input"
        # We implement this by removing the first embedding from the sequence.
        decoder_input = quantized_embedding[:, :, 1:]

        # --- STEP 6: Decoding ---
        # The decoder takes the shifted embeddings and reconstructs the spectrograms
        out_left_stft_ch, out_right_stft_ch = self.decoder(decoder_input)

        out_wav = self.to_waveform(out_left_stft_ch, out_right_stft_ch)
        return out_wav


if __name__ == '__main__':
    model = MeteoSpectroStream()
    x, fs = load_audio_torch(
        "data/preprocessed/fl_3s_avifauna_flamingos_sept25_data/64/er_file_2025_09_07_01_06_05.250000.wav")
    y = model(x.unsqueeze(0))

    zl, zr = model.to_spectrogram(x.unsqueeze(0))
    c = model.encoder(zl, zr)
    

    out_left_stft_ch, out_right_stft_ch = model.decoder(c[:,:,1:])


