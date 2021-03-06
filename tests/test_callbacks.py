import torch
import pytest
import logging
import numpy as np
import torch.nn as nn
import torch.optim as optim

from pathlib import Path
from torch.utils.data import DataLoader
from torchfitter.trainer import Trainer
from torchfitter.utils import DataWrapper
from torchfitter.conventions import ParamsDict
from torchfitter.testing import change_model_params
from sklearn.model_selection import train_test_split
from torchfitter.callbacks.base import CallbackHandler
from torchfitter.callbacks import (
    EarlyStopping, 
    LoggerCallback,
    LearningRateScheduler
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


def test_earlystopping(train_config):
    (
        train_loader,
        val_loader,
        model,
        criterion,
        optimizer,
        device
    ) = train_config

    early_stopping = EarlyStopping(patience=10, load_best=True)
    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        callbacks=[early_stopping]
    )

    params_dict = trainer.params_dict
    early_stopping.on_fit_start(params_dict)

    for i in range(0, 20):
        params_dict[ParamsDict.VAL_LOS] = 1
        params_dict[ParamsDict.EPOCH_NUMBER] = i

        early_stopping.on_epoch_end(params_dict)

        if params_dict[ParamsDict.STOP_TRAINING]:
            break

    msg = "Early stopping not applied at correct epoch."
    assert early_stopping.stopped_epoch == 10, msg

    msg = "Early stopping 'wait' param not correct."
    assert early_stopping.wait == 10, ""


def test_logger_callback(caplog, train_config):
    caplog.set_level(logging.INFO)

    (
        train_loader,
        val_loader,
        model,
        criterion,
        optimizer,
        device
    ) = train_config

    logger = LoggerCallback(update_step=100)

    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        callbacks=[logger]
    )

    trainer.fit(train_loader, val_loader, epochs=10)

    msg = "Logging not in INFO level."
    assert caplog.records[0].levelname == "INFO", msg

    msg = "Logger not stating where the training is running."
    assert caplog.records[0].message == "Starting training process on cpu", msg

    log_msg = "Epoch: 1/10       | Train loss: 8444.365118   | Validation loss: 8672.903764   | Time/epoch:"
    msg = "Logger not logging first epoch correctly"
    assert caplog.records[1].message.startswith(log_msg), msg

    msg = "Logger not logging end of training time"
    assert caplog.records[2].message.startswith("End of training. Total time: "), msg


def test_learning_rate_scheduler(train_config):
    (
        train_loader,
        val_loader,
        model,
        criterion,
        optimizer,
        device
    ) = train_config

    sch = LearningRateScheduler(
        scheduler=optim.lr_scheduler.StepLR(
            optimizer,
            step_size=50,
            gamma=0.9
        )
    )

    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        callbacks=[sch]
    )

    trainer.fit(train_loader, val_loader, epochs=25)

    expected_lr = [
        0.005,
        0.005,
        0.005,
        0.005,
        0.005,
        0.005,
        0.005,
        0.005,
        0.005,
        0.005,
        0.005,
        0.005,
        0.005,
        0.005,
        0.005,
        0.005,
        0.005,
        0.005,
        0.005,
        0.005,
        0.005,
        0.005,
        0.005,
        0.005,
        0.0045000000000000005
    ]

    obtained_lr = trainer.params_dict['history']['learning_rate']

    msg = "Error en LR values"
    expected_lr == obtained_lr, msg