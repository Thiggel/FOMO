from experiment.SSLMethods.IJepa import IJepa

import pytest
import torch
import pytorch_lightning as pl
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from experiment.utils.get_training_args import get_training_args
from experiment.__main__.py import init_model, init_datamodule, init_ssl_type

@pytest.fixture
def model():
    # Define your model initialization here
    args = get_training_args
    checkpoint_filename = (
        args.model_name
        + "_"
        + args.imagenet_variant
        + "_"
        + args.imbalance_method
        + "-{epoch}-{val_loss:.2f}"
        if args.checkpoint is None
        else args.checkpoint
    )

    datamodule = init_datamodule(
        args,
        checkpoint_filename,
    )

    model = init_model(args, datamodule)

    ssl_type = init_ssl_type(args, model, len(datamodule.train_dataloader()))

    return ssl_type

@pytest.fixture
def train_loader():
    # Define your train dataloader initialization here
    return torch.utils.data.DataLoader(torch.randn(10, 10), batch_size=2)

@pytest.fixture
def val_loader():
    # Define your validation dataloader initialization here
    return torch.utils.data.DataLoader(torch.randn(10, 10), batch_size=2)

def test_model_training_step(model, train_loader):
    trainer = pl.Trainer()
    output = trainer.fit(model, train_loader)
    assert output == 1  # Trainer.fit() should return 1 upon successful completion

def test_model_validation_step(model, val_loader):
    trainer = pl.Trainer()
    output = trainer.validate(model, val_loader)
    assert output == 1  # Trainer.validate() should return 1 upon successful completio