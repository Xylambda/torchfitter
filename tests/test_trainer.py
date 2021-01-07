import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim

from pathlib import Path
from torch.utils.data import DataLoader
from torchfitter.trainer import Trainer
from torchfitter.utils import DataWrapper
from sklearn.model_selection import train_test_split

torch.manual_seed(0)
np.random.seed(0)

_path = Path(__file__).parent
DATA_PATH = _path / "data"


def test_trainer():
    model = nn.Linear(in_features=1, out_features=1)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters())

    trainer = Trainer(
        model=model, 
        criterion=criterion,
        optimizer=optimizer, 
        logger_kwargs={'show': True, 'update_step':20}
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

    expected_train_loss = np.array([
        7215.00830078125,
        7211.41943359375,
        7207.84619140625,
        7204.2802734375,
        7200.7197265625,
        7197.1630859375,
        7193.60986328125,
        7190.06005859375,
        7186.5126953125,
        7182.96826171875
    ])

    expected_val_loss = np.array([
        8710.00390625,
        8705.8271484375,
        8701.6640625,
        8697.509765625,
        8693.359375,
        8689.2138671875,
        8685.0732421875,
        8680.9345703125,
        8676.798828125,
        8672.6650390625
    ])

    obtained_train_loss = np.array(trainer.train_loss_)
    obtained_val_loss = np.array(trainer.val_loss_)

    np.testing.assert_almost_equal(obtained_train_loss, expected_train_loss)
    np.testing.assert_almost_equal(obtained_val_loss, expected_val_loss)