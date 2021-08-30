import torch
import pytest
import torch.nn as nn

from torchfitter.testing import (
    change_model_params,
    compute_forward_gradient,
    check_monotonically_decreasing     
)


def test_change_model_params():
    model = nn.Linear(in_features=1, out_features=1)

    weights=torch.Tensor([[-0.065]])
    biases=torch.Tensor([0.5634])

    change_model_params(model, weights, biases)

    expected_weigths = -0.06499999761581421
    expected_biases = 0.5633999705314636

    msg = "Error when assigning new biases."
    assert expected_biases == model.bias.item(), msg

    msg = "Error when assigning new weights."
    assert expected_weigths == model.weight.item(), msg


@pytest.mark.xfail
def test_compute_forward_gradient():
    pass


@pytest.mark.xfail
def test_check_monotonically_decreasing():
    pass