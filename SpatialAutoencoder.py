"""
@File  : SpatialAutoencoder.py
@Date  : 4/3/20269:00 AM
@License: See license file is in the root of the repository.
@Desc  :

Copyright (c) Aki Härmä, DACS, Maastricht University, 2026.
"""

import os
from pathlib import Path
import torch
from torch import optim, nn 
import lightning as L
from SpatialSpectrumDataloader import SpatialDataset, MetaDataNormalizer
from DenseModel import DenseModelEncoder, DenseModelDecoder
from CNN2Model import Cnn2ModelEncoder
from VITModel import ViTEncoder
from sklearn.model_selection import train_test_split

# Optimization for Blackwell
torch.set_float32_matmul_precision('medium')

# define the LightningModule
class SpatialAutoEncoder(L.LightningModule):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        x, m = batch
        z = self.encoder(x)
        zm = torch.cat([z, m], dim=1)
        x_hat = self.decoder(zm)
        loss = nn.functional.mse_loss(x_hat, x)
        # Logging to TensorBoard (if installed) by default
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, m = batch
        z = self.encoder(x)
        zm = torch.cat([z, m], dim=1)
        x_hat = self.decoder(zm)
        loss = nn.functional.mse_loss(x_hat, x)
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


def main():
    files = [str(x) for x in Path("specData2").glob("*.pt")]
    train_files, test_files = train_test_split(files, test_size=0.05)
    # For not to fill the memory
    test_files = test_files[:8]

    train_loader = SpatialDataset(train_files, 32)
    validate_loader = SpatialDataset(test_files,32)

    embed_dim = 2048*4
    Nmeta = len(train_loader.metanorm.meta_labels)
    # encoder = DenseModelEncoder(train_loader.input_shape()[1:], embed_dim)
    # encoder = Cnn2ModelEncoder(train_loader.input_shape()[1:],
    #                           embed_dim-Nmeta)
    encoder = ViTEncoder([400,32], embed_dim-Nmeta)
    decoder = DenseModelDecoder(embed_dim, embed_dim, train_loader.input_shape()[1:])

    checkpoint = "lightning_logs/version_58/checkpoints/epoch=605-step=36336.ckpt"
    # checkpoint = None

    if checkpoint is None:
        # init the autoencoder
        autoencoder = SpatialAutoEncoder(encoder, decoder)
    else:
        autoencoder = SpatialAutoEncoder.load_from_checkpoint(checkpoint_path=checkpoint,
                                                              encoder=encoder,
                                                              decoder=decoder)

    # Compile the model
    # autoencoder = torch.compile(autoencoder)

    trainer = L.Trainer(max_epochs=8000000)
    trainer.fit(model=autoencoder, train_dataloaders=train_loader,
            val_dataloaders=validate_loader)

if __name__ == '__main__':
    main()
