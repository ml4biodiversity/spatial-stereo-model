"""
@File   :   AvexProbeModel.py
@Date   :   26-6-202610:37
@License: See license file in the root of the repository
@Desc   : 

Copyright (c) Aki Härmä, DACS, Maastricht University, 2026.
"""

import pandas as pd
import torch
from pathlib import Path
from torch import optim, nn
from torch.utils.data import Dataset
import librosa
from avex import load_model, list_models, build_model, describe_model, get_model_spec, build_model_from_spec
from avex.configs import ProbeConfig
from avex.models.probes import build_probe_from_config
import lightning as L


# Optimization for Blackwell
torch.set_float32_matmul_precision('medium')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


"""
    Dataset definition
"""
class ZooDataset(Dataset):
    def __init__(self, metadata):
        self.meta = metadata

    def __len__(self):
        return self.meta.shape[0]

    def __getitem__(self, idx):
        sig, sr = librosa.load(self.meta[idx, "audio_file"], sr=16000)
        sig = torch.tensor(sig).unsqueeze(0)
        return sig, self.meta[idx, "target_index"]


# Frozen Avex backbone model and a linear probe
class AvexProbeModel(L.LightningModule):
    model = None
    probe_config = None
    def __init__(self):
        super().__init__()
        self.probe_config = ProbeConfig(
            probe_type="linear",
            target_layers=["backbone"],
            aggregation="mean",
            freeze_backbone=True,
            online_training=True,
        )

    def build_probe(self, model_name, num_classes):
        model_spec = get_model_spec(model_name)
        # Build backbone-only model
        backbone = build_model_from_spec(model_spec, device=device).to(device)
        backbone.eval()

        # Attach a simple linear probe for a 10-class task
        self.model = build_probe_from_config(
            probe_config=self.probe_config,
            base_model=backbone,
            num_classes=num_classes,
            device=device,
        ).to(device)

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        x, yt = batch
        y = self.model(x)
        loss = nn.functional.cross_entropy(yt, y)
        # Logging to TensorBoard (if installed) by default
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, yt = batch
        y = self.model(x)
        loss = nn.functional.cross_entropy(yt, y)
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


if __name__ == '__main__':
    model_name = "esp_aves2_naturelm_audio_v1_beats"
    num_classes = 30
    probe = AvexProbeModel()
    probe.build_probe(model_name, num_classes)

    files = list(Path("./data/").glob("*.xlmx"))


