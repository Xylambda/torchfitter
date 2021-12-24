import torch
import pytest
import numpy as np
import torch.nn as nn
import torch.optim as optim

from pathlib import Path
from torch.utils.data import DataLoader
from torchfitter.trainer import Trainer
from torchfitter.utils.data import DataWrapper
from torchfitter.testing import change_model_params

from sklearn.model_selection import train_test_split


@pytest.mark.xfail
def test_manager():
    pass
