import os.path
import hydra
import numpy as np
import torch
from hydra.core.config_store import ConfigStore

from nanogpt.tokenizer import CharacterTokenizer
from ..transformer_module import TransformerTrainingConfig, TransformerModule
import lightning as L
from lightning.pytorch import loggers as pl_loggers


cs = ConfigStore.instance()
cs.store(name="config", node=TransformerTrainingConfig)


@hydra.main(config_name="config", config_path=None, version_base='1.3.2')
def main(config: TransformerTrainingConfig):
    this_dir = os.path.abspath(os.path.dirname(__file__))
    data = torch.load(os.path.join(this_dir, 'data/shakespeare.pt'))
    encoder = CharacterTokenizer.load_from_file(os.path.join(this_dir, 'data/encoder_characters.txt'))

    model = TransformerModule(config, data, encoder)

    # setup wandb logger
    wandb_config = {
        'project': 'nanogpt',
        'group': 'shakespeare'
    }
    wandb_logger = pl_loggers.WandbLogger(**wandb_config)

    trainer = L.Trainer(max_epochs=1, logger=wandb_logger, log_every_n_steps=1)
    trainer.fit(model)


if __name__ == '__main__':
    main()
