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
from torchfitter.utils import DataWrapper
from sklearn.model_selection import train_test_split
from torchfitter.regularization import L1Regularization
from torchfitter.callbacks import (
    LoggerCallback,
    EarlyStopping,
    LearningRateScheduler
)

# -----------------------------------------------------------------------------
torch.manual_seed(0)
np.random.seed(0)

DATA_PATH = Path(os.path.abspath('')).parent / "tests/data"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------------------------------------------------------------
X = np.load(DATA_PATH / "features.npy")
y = np.load(DATA_PATH / "labels.npy")
y = y.reshape(-1,1)


# simplest case of cross-validation
X_train, X_val, y_train, y_val = train_test_split(
    X, 
    y, 
    test_size=0.33, 
    random_state=42
)

# -----------------------------------------------------------------------------
model = nn.Linear(in_features=1, out_features=1)
model.to(DEVICE)

regularizer = L1Regularization(regularization_rate=0.01, biases=False)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.005)

callbacks = [
    LoggerCallback(update_step=100),
    EarlyStopping(patience=100, load_best=True),
    LearningRateScheduler(
        scheduler=optim.lr_scheduler.StepLR(
            optimizer, step_size=500, gamma=0.9
        )
    )
]

metrics = [
    torchmetrics.MeanSquaredError().to(DEVICE),
    torchmetrics.MeanAbsoluteError().to(DEVICE)
]

# -----------------------------------------------------------------------------
# wrap data in Dataset
train_wrapper = DataWrapper(
    X_train, 
    y_train, 
    dtype_X='float', 
    dtype_y='float'
)

val_wrapper = DataWrapper(
    X_val,
    y_val,
    dtype_X='float',
    dtype_y='float'
)

# torch Loaders
train_loader = DataLoader(train_wrapper, batch_size=64, pin_memory=True)
val_loader = DataLoader(val_wrapper, batch_size=64, pin_memory=True)

# -----------------------------------------------------------------------------
trainer = Trainer(
    model=model, 
    criterion=criterion,
    optimizer=optimizer, 
    regularizer=regularizer,
    device=DEVICE,
    callbacks=callbacks,
    metrics=metrics
)

if __name__ == "__main__":
    # -------------------------------------------------------------------------
    # argument parsing
    parser = argparse.ArgumentParser('')
    parser.add_argument('--epochs', type=int, default=5000)
    
    args = parser.parse_args()
    n_epochs = args.epochs

    # -------------------------------------------------------------------------
    # fitting process
    trainer.fit(train_loader, val_loader, epochs=n_epochs)

    # predictions
    with torch.no_grad():
        to_predict = torch.from_numpy(X_val).float().to(DEVICE)
        y_pred = model(to_predict).cpu().numpy()

    # -------------------------------------------------------------------------
    # plot predictions, losses and learning rate
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(19,4))

    ax[0].plot(trainer.params_dict['history']['train_loss'], label='Train loss')
    ax[0].plot(trainer.params_dict['history']['validation_loss'], label='Validation loss')
    ax[0].set_title('Train and validation losses')
    ax[0].legend();

    ax[1].plot(X_val, y_val, '.', label="Real")
    ax[1].plot(X_val, y_pred, '.', label="Prediction")
    ax[1].set_title('Predictions')
    ax[1].legend();


    ax[2].plot(trainer.params_dict['history']['learning_rate'], label="Learning rate")
    ax[2].set_title('Learning Rate')
    ax[2].legend();
    plt.show()

    # plot metrics evolution
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15,5))

    ax[0].plot(trainer.internal_state.history['MeanSquaredError']['train'], label='Train')
    ax[0].plot(trainer.internal_state.history['MeanSquaredError']['validation'], label='Validation')
    ax[0].set_title('Mean Squared Error')
    ax[0].legend();

    ax[1].plot(trainer.internal_state.history['MeanAbsoluteError']['train'], label='Train')
    ax[1].plot(trainer.internal_state.history['MeanAbsoluteError']['validation'], label='Validation')
    ax[1].set_title('Mean Absolute Error')
    ax[1].legend();
    plt.show()