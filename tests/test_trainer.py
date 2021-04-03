import torch
import pytest
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
        device=device
    )
    
    # fitting process
    trainer.fit(train_loader, val_loader, epochs=10)

    expected_train_loss = np.array(
        [
            8444.36511811756,
            8440.451636904761,
            8436.565499441964,
            8432.687313988095,
            8428.814755394345,
            8424.945545014882,
            8421.079427083334,
            8417.215611049107,
            8413.354399181548,
            8409.495140438989
        ]
    )

    expected_val_loss = np.array(
        [
            8672.903764204546,
            8668.988059303978,
            8665.083984375,
            8661.185191761364,
            8657.291459517046,
            8653.400834517046,
            8649.512428977272,
            8645.626642400568,
            8641.742720170454,
            8637.861061789772
        ]
    )

    obtained_train_loss = np.array(
        trainer.params_dict['history']['train_loss']
    )
    obtained_val_loss = np.array(
        trainer.params_dict['history']['validation_loss']
    )

    np.testing.assert_almost_equal(obtained_train_loss, expected_train_loss)
    np.testing.assert_almost_equal(obtained_val_loss, expected_val_loss)