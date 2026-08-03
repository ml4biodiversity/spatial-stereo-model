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
from sklearn.model_selection import train_test_split


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
        sig, sr = librosa.load(self.meta.loc[idx, "audio_file"], sr=16000)
        sig = torch.tensor(sig).unsqueeze(0)
        return sig, int(self.meta.loc[idx, "target_index"])


# Frozen Avex backbone model and a linear probe
class AvexProbeModel(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.probe_config = ProbeConfig(
            probe_type="linear",
            target_layers=["backbone"],
            aggregation="",
            freeze_backbone=True,
            online_training=True,
        )
        self.loss_function = torch.nn.CrossEntropyLoss()

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
        loss = self.loss_function(y, torch.LongTensor([yt]).to(device))
        # Logging to TensorBoard (if installed) by default
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, yt = batch
        y = self.model(x)
        loss = self.loss_function(y,torch.LongTensor([yt]).to(device))
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


if __name__ == '__main__':
    model_name = "esp_aves2_naturelm_audio_v1_beats"
    dpath = "/media/kakskyt/data/zoodata/er_path/"
    files = list(Path(dpath).glob("*flami*"))

    metafiles = [f"{str(f)}/{str(f).split("/")[-1]}_metadata.xlsx" for f in files]
    num_classes = len(metafiles)
    probe = AvexProbeModel()
    probe.build_probe(model_name, num_classes)

    meta = pd.read_excel(metafiles[0], index_col=0)
    for f in metafiles[1:]:
        meta = pd.concat([meta, pd.read_excel(f, index_col=0)], ignore_index=True)

    names = [str(x).split("/")[-1] for x in files]
    labelmap = {names[c1]:c1 for c1 in range(len(names))}
    # Add fields
    meta["audio_file"] = dpath + meta["filename"]
    meta["target_name"] = meta["filename"].apply(lambda x: str(x).split("/")[0])
    meta["target_index"] = meta["target_name"].apply(lambda x: labelmap[x])

    train_files, test_files = train_test_split(meta, test_size=0.05)
    train_files = train_files.reset_index(drop=True)
    test_files = test_files.reset_index(drop=True)
    train_dataset = ZooDataset(train_files)
    test_dataset = ZooDataset(test_files)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
    )
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
    )

    trainer = L.Trainer(max_epochs=8000)
    trainer.fit(model=probe, train_dataloaders=train_dataloader,
            val_dataloaders=test_dataloader)