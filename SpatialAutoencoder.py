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
from SpatialSpectrumDataloader import SpatialDataset
from DenseModel import DenseModelEncoder, DenseModelDecoder
from CNN2Model import Cnn2ModelEncoder
from sklearn.model_selection import train_test_split

# define the LightningModule
class SpatialAutoEncoder(L.LightningModule):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        x, _ = batch
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = nn.functional.mse_loss(x_hat, x)
        # Logging to TensorBoard (if installed) by default
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, _ = batch
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = nn.functional.mse_loss(x_hat, x)
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


def main():
    files = [str(x) for x in Path("specData").glob("*.pt")]
    train_files, test_files = train_test_split(files[:50], test_size=0.2)

    train_loader = SpatialDataset(train_files, 16, packets=True)
    validate_loader = SpatialDataset(test_files,16)

    # init the autoencoder
    embed_dim = 1024
    # encoder = DenseModelEncoder(train_loader.input_shape()[1:], embed_dim)
    encoder = Cnn2ModelEncoder(train_loader.input_shape()[1:], embed_dim)
    decoder = DenseModelDecoder(embed_dim, train_loader.input_shape()[1:])
    autoencoder = SpatialAutoEncoder(encoder, decoder)

    # Compile the model
    autoencoder = torch.compile(autoencoder)

    trainer = L.Trainer(max_epochs=10)
    trainer.fit(model=autoencoder, train_dataloaders=train_loader,
            val_dataloaders=validate_loader)

if __name__ == '__main__':
    main()
