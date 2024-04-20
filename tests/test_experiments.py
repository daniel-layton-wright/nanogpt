import unittest
import torch.utils.data
import numpy as np
import lightning as L
import torch.testing


class TestExperiments(unittest.TestCase):
    def test_dataset(self):
        from nanogpt.experiments.transformer_module import TransformerDataset, TransformerConfig

        data = np.array([1, 5, 10, 4, 3])
        config = TransformerConfig(context_size=2)

        dataset = TransformerDataset(data, config)

        expected_x = torch.LongTensor([[1, 5], [5, 10], [10, 4]])
        expected_y = torch.LongTensor([[5, 10], [10, 4], [4, 3]])

        for i in range(len(dataset)):
            x, y = dataset[i]
            torch.testing.assert_close(x, expected_x[i])
            torch.testing.assert_close(y, expected_y[i])

    def test_transformer_module(self):
        from nanogpt.experiments.transformer_module import TransformerModule, TransformerTrainingConfig

        config = TransformerTrainingConfig(vocab_size=10, context_size=2, embed_dim=4, num_blocks=2)
        module = TransformerModule(config)

        loss = module.loss(torch.LongTensor([[1, 2], [3, 4]]), torch.LongTensor([[1, 2], [3, 4]]))

    def test_training(self):
        from nanogpt.experiments.transformer_module import (TransformerModule, TransformerTrainingConfig,
                                                            TransformerDataset)

        config = TransformerTrainingConfig(vocab_size=11, context_size=2, embed_dim=4, num_blocks=2)
        model = TransformerModule(config)

        data = np.array([1, 5, 10, 4, 3])
        train_data = TransformerDataset(data, config)
        val_data = TransformerDataset(data, config)

        # Wrap train data in a dataloader with batch size 2
        train_data = torch.utils.data.DataLoader(train_data, batch_size=2)
        val_data = torch.utils.data.DataLoader(val_data, batch_size=2)

        trainer = L.Trainer(max_epochs=1)
        trainer.fit(model, train_data, val_data)

    def test_training_with_encapsulated_dataloader(self):
        from nanogpt.experiments.transformer_module import (TransformerModule, TransformerTrainingConfig,
                                                            TransformerDataset)

        config = TransformerTrainingConfig(vocab_size=11, context_size=2, embed_dim=4, num_blocks=2,
                                           train_data_fraction=0.8)
        data = np.array([1, 5, 10, 4, 3, 2, 1])
        model = TransformerModule(config, data=data)
        trainer = L.Trainer(max_epochs=1)
        trainer.fit(model)
