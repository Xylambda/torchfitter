"""
In this example, a regression model with the ability to predict a mean and
standard deviation is created and trained using torchfitter.

By predicting a mean and a std. one can define some sort of uncertainty
interval around the predictions (a.k.a. how sure is my model about the
prediction of this sample?).
"""

import torch
import argparse
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torchfitter.conventions import ParamsDict
from sklearn.datasets import make_regression
from torchfitter.utils.preprocessing import train_test_val_split
from torchfitter.trainer import Trainer
from torch.utils.data import DataLoader
from torchfitter.utils.data import DataWrapper
from torchfitter.callbacks import RichProgressBar, EarlyStopping


class DeepNormal(nn.Module):
    """Neural network with parametrizable normal distribution as output.

    Taken from [1].
    
    References
    ----------
    .. [1] Romain Strock - Modeling uncertainty with Pytorch:
       https://romainstrock.com/blog/modeling-uncertainty-with-pytorch.html
    """
    def __init__(self, n_inputs, n_hidden):
        super().__init__()

        # Shared parameters
        self.shared_layer = nn.Sequential(
            nn.Linear(n_inputs, n_hidden),
            nn.ReLU(),
            nn.Dropout(),
        )
        
        # Mean parameters
        self.mean_layer = nn.Sequential(
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(n_hidden, 1),
        )
        
        # Standard deviation parameters
        self.std_layer = nn.Sequential(
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(n_hidden, 1),
            nn.Softplus(),  # enforces positivity
        )
             
    def forward(self, x):
        # Shared embedding
        shared = self.shared_layer(x)
        
        # Parametrization of the mean
        μ = self.mean_layer(shared)
        
        # Parametrization of the standard deviation
        σ = self.std_layer(shared)
        
        return torch.distributions.Normal(μ, σ)


class NLLLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, output, target):
        """
        Assumes `output` is a distribution.
        """
        neg_log_likelihood = -output.log_prob(target)
        return torch.mean(neg_log_likelihood)


def main():
    # -------------------------------------------------------------------------
    # argument parsing
    parser = argparse.ArgumentParser("")
    parser.add_argument("--epochs", type=int, default=5000)

    args = parser.parse_args()
    n_epochs = args.epochs

    # -------------------------------------------------------------------------
    # generate dummy data
    X, y = make_regression(
        n_samples=5000, n_features=1, n_informative=1, noise=5, random_state=0
    )
    y = y.reshape(-1,1)

    # split data into train, test and validation
    _tup = train_test_val_split(X, y)
    X_train, y_train, X_val, y_val, X_test, y_test = _tup

    # wrap data in Dataset
    train_wrapper = DataWrapper(
        X_train, y_train, dtype_X="float", dtype_y="float"
    )
    val_wrapper = DataWrapper(X_val, y_val, dtype_X="float", dtype_y="float")

    # torch Loaders
    train_loader = DataLoader(train_wrapper, batch_size=64, pin_memory=True)
    val_loader = DataLoader(val_wrapper, batch_size=64, pin_memory=True)

    # -------------------------------------------------------------------------
    # define model, optimizer and loss
    criterion = NLLLoss()
    model = DeepNormal(n_inputs=X.shape[1], n_hidden=15)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)

    # callbacks list
    callbacks = [
        EarlyStopping(patience=150, load_best=True),
        RichProgressBar(display_step=50)
    ]

    # instantiate Trainer object with all the configuration
    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        callbacks=callbacks,
    )

    # train process
    history = trainer.fit(train_loader, val_loader, epochs=n_epochs)

    # -------------------------------------------------------------------------
    # this is a torch distribution
    distr_prediction = trainer.predict(X_test)

    # get mean and standard deviation for each sample in test
    y_pred = distr_prediction.mean
    y_pred_std = distr_prediction.stddev

    # to array
    try:
        y_pred = y_pred.cpu().detach().numpy().flatten()
        y_pred_std = y_pred_std.cpu().detach().numpy().flatten()
    except:
        y_pred = y_pred.detach().numpy().flatten()
        y_pred_std = y_pred_std.detach().numpy().flatten()

    # -------------------------------------------------------------------------
    # plot losses, mean predictions and lr
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(19, 4))
    epoch_hist = history[ParamsDict.EPOCH_HISTORY]

    ax[0].plot(epoch_hist[ParamsDict.LOSS]["train"], label="Train loss")
    ax[0].plot(
        epoch_hist[ParamsDict.LOSS]["validation"], label="Validation loss"
    )
    ax[0].set_title("Train and validation losses")
    ax[0].grid()
    ax[0].legend()

    ax[1].plot(X_test, y_test, ".", label="Real")
    ax[1].plot(X_test, y_pred, ".", label="Prediction")
    ax[1].set_title("Predictions")
    ax[1].grid()
    ax[1].legend()

    ax[2].plot(epoch_hist[ParamsDict.HISTORY_LR], label="Learning rate")
    ax[2].set_title("Learning Rate")
    ax[2].legend()
    ax[2].grid()
    plt.show()

    # -------------------------------------------------------------------------
    # create some upper and lower bounds
    lower = y_pred - 2 * y_pred_std
    upper = y_pred + 2 * y_pred_std

    fig, ax = plt.subplots(1, 1, figsize=(15,8))

    ax.plot(X_test, y_test, "*k")
    ax.scatter(X_test.flatten(), y_pred, label="predicted means")

    ax.scatter(X_test.flatten(), lower)
    ax.scatter(X_test.flatten(), upper)

    ax.grid(True)
    ax.legend()


if __name__ == "__main__":
    main()