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
from sklearn.model_selection import train_test_split
from torchfitter.callbacks.base import CallbackHandler
from torchfitter.testing import (
    change_model_params, 
    check_monotonically_decreasing
)
from torchfitter.callbacks import (
    EarlyStopping, 
    LoggerCallback,
    LearningRateScheduler,
    ReduceLROnPlateau,
    GPUStats,
    RichProgressBar
)

from torchfitter.callbacks.base import CallbackHandler, Callback

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

    return train_loader, val_loader, model, criterion, optimizer


def test_earlystopping(train_config):
    (
        _,
        _,
        model,
        criterion,
        optimizer,
    ) = train_config

    early_stopping = EarlyStopping(patience=10, load_best=True)
    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        callbacks=[early_stopping]
    )

    params_dict = trainer.internal_state.get_state_dict()
    early_stopping.on_fit_start(params_dict)

    for i in range(0, 20):
        params_dict[ParamsDict.VAL_LOSS] = 1
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
    ) = train_config

    logger = LoggerCallback(update_step=100)

    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        callbacks=[logger]
    )

    trainer.fit(train_loader, val_loader, epochs=10)

    msg = "Logging not in INFO level."
    assert caplog.records[0].levelname == "INFO", msg

    msg = "Logger not stating where the training is running."
    assert caplog.records[0].message.startswith("Starting training process"), msg

    log_msg = "Epoch 1/10 | "
    msg = "Logger not logging first epoch correctly"
    assert caplog.records[1].message.startswith(log_msg), msg
    assert "| Validation loss" in caplog.records[1].message, msg
    assert "| Time/epoch:" in caplog.records[1].message, msg

    msg = "Logger not logging end of training time"
    assert caplog.records[2].message.startswith("End of training. Total time: "), msg


@pytest.mark.xfail
def test_progress_bar_logger():
    pass


def test_learning_rate_scheduler(train_config):
    (
        train_loader,
        val_loader,
        model,
        criterion,
        optimizer,
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
        callbacks=[sch]
    )

    trainer.fit(train_loader, val_loader, epochs=25)
    obtained_lr = trainer.internal_state.get_state_dict()[ParamsDict.EPOCH_HISTORY]['learning_rate']

    msg = "LR values are not monotonically decreasing."
    assert check_monotonically_decreasing(obtained_lr), msg


@pytest.mark.xfail
def test_reduce_lr_on_plateau():
    pass


@pytest.mark.xfail
def test_gpu_stats():
    pass


@pytest.mark.xfail
def test_rich_progress_bar():
    pass


@pytest.mark.fail
def test_callback_handler():

    class CallbackTester(Callback):
        def __init__(self) -> None:
            super(CallbackTester, self).__init__()

        def on_train_step_start(self, params_dict: dict) -> str:
            expected = "on_train_step_start"

        def on_train_step_end(self, params_dict: dict) -> str:
            expected = "on_train_step_end"

        def on_validation_step_start(self, params_dict: dict) -> str:
            expected = "on_validation_step_start"

        def on_validation_step_end(self, params_dict: dict) -> str:
            expected = "on_validation_step_end"

        def on_epoch_start(self, params_dict: dict) -> str:
            expected = "on_epoch_start"

        def on_epoch_end(self, params_dict: dict) -> str:
            expected = "on_epoch_end"

        def on_fit_start(self, params_dict: dict) -> str:
            expected = "on_fit_start"

        def on_fit_end(self, params_dict: dict) -> str:
            expected = "on_fit_end"


    # -------------------------------------------------------------------------
    callback = CallbackTester()
    handler = CallbackHandler([callback])
