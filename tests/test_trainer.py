import torch
import pytest
import numpy as np
import torch.nn as nn
import torch.optim as optim

from pathlib import Path
from torch.utils.data import DataLoader
from torchfitter.trainer import Trainer
from torchfitter.utils import DataWrapper
from torchfitter.conventions import ParamsDict
from sklearn.model_selection import train_test_split
from torchfitter.testing import (
    change_model_params, 
    check_monotonically_decreasing
)

torch.manual_seed(0)
np.random.seed(0)

_path = Path(__file__).parent
DATA_PATH = _path / "data"


@pytest.fixture
def train_config():
    # we create a model and set known params
    model = nn.Linear(in_features=1, out_features=1)
    change_model_params(
        model, 
        weights=torch.Tensor([[-0.065]]), 
        biases=torch.Tensor([0.5634])
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters())

    X = np.load(DATA_PATH / "features.npy")
    y = np.load(DATA_PATH / "labels.npy")
    
    y = y.reshape(-1,1)
    
    # we don't need X_test, y_test to check the Trainer works
    X_train, X_val, y_train, y_val = train_test_split(
        X, 
        y, 
        test_size=0.33, 
        random_state=42
    )
    
    # wrap data in Dataset
    train_wrapper = DataWrapper(
        X_train, 
        y_train, 
        dtype_X='float', 
        dtype_y='float'
    )
    val_wrapper = DataWrapper(X_val, y_val, dtype_X='float', dtype_y='float')
    
    # torch Loaders
    train_loader = DataLoader(train_wrapper, batch_size=32)
    val_loader = DataLoader(val_wrapper, batch_size=32)

    return train_loader, val_loader, model, criterion, optimizer, device


def test_trainer(train_config):

    (
        train_loader,
        val_loader,
        model,
        criterion,
        optimizer,
        device
    ) = train_config

    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        mixed_precision=False
    )
    
    # fitting process
    trainer.fit(train_loader, val_loader, epochs=100)

    obtained_train_loss = np.array(
        trainer.internal_state.get_state_dict()[ParamsDict.HISTORY][ParamsDict.HISTORY_TRAIN_LOSS]
    )
    obtained_val_loss = np.array(
        trainer.internal_state.get_state_dict()[ParamsDict.HISTORY][ParamsDict.HISTORY_VAL_LOSS]
    )

    msg = "Train loss did not strictly decrease"
    assert check_monotonically_decreasing(obtained_train_loss, strict=True), msg

    msg = "Validation loss did not strictly decrease"
    assert check_monotonically_decreasing(obtained_val_loss, strict=True), msg


def test_trainer_mixed_precision(train_config):

    (
        train_loader,
        val_loader,
        model,
        criterion,
        optimizer,
        device
    ) = train_config

    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        mixed_precision=True
    )
    
    # fitting process
    trainer.fit(train_loader, val_loader, epochs=100)

    obtained_train_loss = np.array(
        trainer.internal_state.get_state_dict()[ParamsDict.HISTORY][ParamsDict.HISTORY_TRAIN_LOSS]
    )
    obtained_val_loss = np.array(
        trainer.internal_state.get_state_dict()[ParamsDict.HISTORY][ParamsDict.HISTORY_VAL_LOSS]
    )

    msg = "Train loss did not strictly decrease"
    assert check_monotonically_decreasing(obtained_train_loss, strict=True), msg

    msg = "Validation loss did not strictly decrease"
    assert check_monotonically_decreasing(obtained_val_loss, strict=True), msg
