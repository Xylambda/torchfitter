import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim

from pathlib import Path
from torch.utils.data import DataLoader
from torchfitter.trainer import Trainer
from torchfitter.utils import DataWrapper
from torchfitter.testing import change_model_params

from sklearn.model_selection import train_test_split

torch.manual_seed(0)
np.random.seed(0)

_path = Path(__file__).parent
DATA_PATH = _path / "data"


def test_trainer():
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

    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        logger_kwargs={'show': True, 'update_step':20},
        device=device
    )
    
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
    
    # fitting process
    trainer.fit(train_loader, val_loader, epochs=10)

    expected_train_loss = np.array(
        [
            7225.11279296875,
            7221.5205078125,
            7217.94189453125,
            7214.37158203125,
            7210.80712890625,
            7207.24609375,
            7203.68896484375,
            7200.134765625,
            7196.58251953125,
            7193.03369140625
        ]
    )

    expected_val_loss = np.array(
        [
            8721.7216796875,
            8717.5390625,
            8713.3740234375,
            8709.2138671875,
            8705.060546875,
            8700.9111328125,
            8696.765625,
            8692.6220703125,
            8688.4814453125,
            8684.34375
        ]
    )

    obtained_train_loss = np.array(trainer.train_loss_)
    obtained_val_loss = np.array(trainer.val_loss_)

    np.testing.assert_almost_equal(obtained_train_loss, expected_train_loss)
    np.testing.assert_almost_equal(obtained_val_loss, expected_val_loss)