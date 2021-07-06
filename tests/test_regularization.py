import torch
import pytest
import torch.nn as nn

from torchfitter.regularization import (
    L1Regularization,
    L2Regularization,
)

from torchfitter.testing import change_model_params


@pytest.fixture
def model_config():
    model = nn.Linear(2,2)

    # change weights and biases
    weights = torch.Tensor([[0.5675, 0.8352], [0.2056, 0.5932]]).float()
    biases = torch.Tensor([-0.2723,  0.1896]).float()

    change_model_params(model, weights, biases)

    return model


def test_L1Regularization(model_config):
    model, device = model_config()
    regularizer = L1Regularization(regularization_rate=0.01, biases=False)
    
    obtained_term = regularizer(model.named_parameters(), device='gpu').item()
    expected_term = 0.022014999762177467
    
    msg = "Error in L1 regularization penalty"
    assert obtained_term == expected_term, msg


def test_L2Regularization(model_config):
    model, device = model_config()
    regularizer = L2Regularization(regularization_rate=0.01, biases=False)
    
    obtained_term = regularizer(model.named_parameters()).item()
    expected_term = 0.011890217661857605
    
    msg = "Error in L2 regularization penalty"
    assert obtained_term == expected_term, msg
    