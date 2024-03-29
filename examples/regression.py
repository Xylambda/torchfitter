import os
import torch
import torchmetrics
import argparse
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from pathlib import Path
from torch.utils.data import DataLoader
from torchfitter.trainer import Trainer
from torch.optim.swa_utils import SWALR
from torchfitter.utils.data import DataWrapper
from torchfitter.conventions import ParamsDict
from sklearn.model_selection import train_test_split
from torchfitter.callbacks import (
    EarlyStopping,
    RichProgressBar,
    StochasticWeightAveraging,
    L1Regularization
)

# -----------------------------------------------------------------------------
torch.manual_seed(0)
np.random.seed(0)

DATA_PATH = Path(os.path.abspath("")).parent / "tests/data"


def main():
    # -------------------------------------------------------------------------
    # argument parsing
    parser = argparse.ArgumentParser("")
    parser.add_argument("--epochs", type=int, default=5000)

    args = parser.parse_args()
    n_epochs = args.epochs

    # -------------------------------------------------------------------------
    X = np.load(DATA_PATH / "features.npy")
    y = np.load(DATA_PATH / "labels.npy")
    y = y.reshape(-1, 1)

    # simplest case of cross-validation
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.33, random_state=42
    )

    # -------------------------------------------------------------------------
    model = nn.Linear(in_features=1, out_features=1)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.005)

    # stochastic weight averaging + "standard" scheduler
    swa_sch = SWALR(optimizer, swa_lr=0.05)
    sch = optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.9)
    swa_callback = StochasticWeightAveraging(
        swa_scheduler=swa_sch, start_epoch=150, scheduler=sch
    )

    callbacks = [
        EarlyStopping(patience=100, load_best=True),
        swa_callback,
        RichProgressBar(display_step=100, log_lr=False),
        L1Regularization(regularization_rate=0.01, biases=False)
    ]

    metrics = [
        torchmetrics.MeanSquaredError(), torchmetrics.MeanAbsoluteError()
    ]

    # -------------------------------------------------------------------------
    # wrap data in Dataset
    train_wrapper = DataWrapper(
        X_train, y_train, dtype_X="float", dtype_y="float"
    )
    val_wrapper = DataWrapper(X_val, y_val, dtype_X="float", dtype_y="float")

    # torch Loaders
    train_loader = DataLoader(train_wrapper, batch_size=64, pin_memory=True)
    val_loader = DataLoader(val_wrapper, batch_size=64, pin_memory=True)

    # -------------------------------------------------------------------------
    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        callbacks=callbacks,
        metrics=metrics,
    )

    # -------------------------------------------------------------------------
    # fitting process and predictions
    history = trainer.fit(train_loader, val_loader, epochs=n_epochs)
    y_pred = trainer.predict(X_val, as_array=True)

    # -------------------------------------------------------------------------
    # plot predictions, losses and learning rate
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(19, 4))
    epoch_hist = history[ParamsDict.EPOCH_HISTORY]

    ax[0].plot(epoch_hist[ParamsDict.LOSS]["train"], label="Train loss")
    ax[0].plot(
        epoch_hist[ParamsDict.LOSS]["validation"], label="Validation loss"
    )
    ax[0].set_title("Train and validation losses")
    ax[0].grid()
    ax[0].legend()

    ax[1].plot(X_val, y_val, ".", label="Real")
    ax[1].plot(X_val, y_pred, ".", label="Prediction")
    ax[1].set_title("Predictions")
    ax[1].grid()
    ax[1].legend()

    ax[2].plot(epoch_hist[ParamsDict.HISTORY_LR], label="Learning rate")
    ax[2].set_title("Learning Rate")
    ax[2].legend()
    ax[2].grid()
    plt.show()

    # plot metrics evolution
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))

    mae_hist = history[ParamsDict.EPOCH_HISTORY]["MeanAbsoluteError"]
    mse_hist = history[ParamsDict.EPOCH_HISTORY]["MeanSquaredError"]

    ax[0].plot(mse_hist["train"], label="Train")
    ax[0].plot(mse_hist["validation"], label="Validation")
    ax[0].set_title("Mean Squared Error")
    ax[0].grid()
    ax[0].legend()

    ax[1].plot(mae_hist["train"], label="Train")
    ax[1].plot(mae_hist["validation"], label="Validation")
    ax[1].set_title("Mean Absolute Error")
    ax[1].grid()
    ax[1].legend()
    plt.show()


if __name__ == "__main__":
    main()
