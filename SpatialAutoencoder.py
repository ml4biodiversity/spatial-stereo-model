"""
@File  : SpatialAutoencoder.py
@Date  : 4/3/20269:00 AM
@License: See license file is in the root of the repository.
@Desc  :

Copyright (c) Aki Härmä, DACS, Maastricht University, 2026.
"""

import os
import torch
from torch import optim, nn, utils, Tensor
import lightning as L
from SpatialSpectrumDataloader import SpatialDataset
from DenseModel import DenseModelEncoder, DenseModelDecoder
from CNN2Model import Cnn2ModelEncoder

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

dataset = torch.load("specData/spec_fl_zoo_parc_aug25_data_0.pt",
                     weights_only=False)
train_loader = SpatialDataset(dataset,16)

dataset = torch.load("specData/spec_fl_zoo_parc_aug25_data_1.pt",
                     weights_only=False)
validate_loader = SpatialDataset(dataset,16)

# init the autoencoder
embed_dim = 1024
# encoder = DenseModelEncoder(train_loader.input_shape()[1:], embed_dim)
encoder = Cnn2ModelEncoder(train_loader.input_shape()[1:], embed_dim)
decoder = DenseModelDecoder(embed_dim, train_loader.input_shape()[1:])
autoencoder = SpatialAutoEncoder(encoder, decoder)

# Compile the model
autoencoder = torch.compile(autoencoder)

trainer = L.Trainer(limit_train_batches=208, max_epochs=20)
trainer.fit(model=autoencoder, train_dataloaders=train_loader,
            val_dataloaders=validate_loader)

if __name__ == '__main__':
    pass
