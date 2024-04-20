from dataclasses import dataclass
from typing import Any, Union

import numpy as np
import lightning as L
import torch
import torch.utils.data
from lightning.pytorch.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS
from pytorch_lightning.utilities.types import STEP_OUTPUT

from ..transformer import TransformerConfig, Transformer
import wandb


@dataclass
class TransformerTrainingConfig(TransformerConfig):
    lr: float = 3e-4
    weight_decay = 1e-5
    train_data_fraction: float = 0.8
    batch_size: int = 64
    data_loader_workers: int = 4
    example_text_every_n_iters: int = 1000


class TransformerModule(L.LightningModule):
    def __init__(self, config: TransformerTrainingConfig, data: Union[np.array, None] = None, encoder=None):
        super().__init__()
        self.config = config
        self.model = Transformer(config)
        self.data = data
        self.train_data = None
        self.val_data = None
        self.encoder = encoder

        self.setup_data()

        self.examples_table = []

    def setup_data(self):
        if self.data is None:
            return

        train_size = int(len(self.data) * self.config.train_data_fraction)
        train_data = self.data[:train_size]
        val_data = self.data[train_size:]

        train_data = TransformerDataset(train_data, self.config)
        val_data = TransformerDataset(val_data, self.config)

        self.train_data = torch.utils.data.DataLoader(train_data, batch_size=self.config.batch_size, shuffle=True,
                                                      num_workers=self.config.data_loader_workers)
        self.val_data = torch.utils.data.DataLoader(val_data, batch_size=self.config.batch_size, shuffle=True,
                                                    num_workers=self.config.data_loader_workers)

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.config.lr, weight_decay=self.config.weight_decay)

    def forward(self, x):
        return self.model(x)

    def loss(self, x, y):
        return torch.nn.functional.cross_entropy(self.model(x).view(-1, self.config.vocab_size), y.view(-1))

    def training_step(self, train_batch, batch_idx) -> STEP_OUTPUT:
        x, y = train_batch
        loss = self.loss(x, y)
        self.log('loss', loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        loss = self.loss(x, y)
        self.log('val_loss', loss)

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        if self.train_data is not None:
            return self.train_data

    def val_dataloader(self) -> EVAL_DATALOADERS:
        if self.val_data is not None:
            return self.val_data

    def on_train_batch_end(self, outputs: STEP_OUTPUT, batch: Any, batch_idx: int) -> None:
        if self.global_step % self.config.example_text_every_n_iters == 0:
            text = self.generate_text(100)
            self.examples_table.append([self.global_step, text])
            self.trainer.logger.experiment.log({'examples': wandb.Table(columns=['step', 'text'], data=self.examples_table)})

    def generate_text(self, n_tokens):
        tokens = self.encoder.encode('\n').reshape(1, 1)
        # Move tokens to the right device
        tokens = tokens.to(next(self.model.parameters()).device)
        tokens = self.model.generate(tokens, n_tokens, temperature=1.0)
        text = self.encoder.decode(tokens.flatten())
        return text


class TransformerDataset(torch.utils.data.Dataset):
    def __init__(self, data: Union[torch.Tensor, np.ndarray], config: TransformerConfig, dtype=torch.long):
        self.data = torch.tensor(data, dtype=dtype)
        self.context_size = config.context_size

    def __len__(self):
        return len(self.data) - self.context_size

    def __getitem__(self, idx) -> Any:
        return self.data[idx:idx + self.context_size], self.data[idx+1:idx+self.context_size+1]
