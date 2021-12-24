import torch
import pytest
from torch._C import device
import torch.nn as nn

from torchfitter.regularization import (
    L1Regularization,
    L2Regularization,
    ElasticNetRegularization,
)

from torchfitter.testing import change_model_params


@pytest.fixture
def model_config():
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    model = nn.Linear(2, 2)

    # change weights and biases
    weights = torch.Tensor([[0.5675, 0.8352], [0.2056, 0.5932]]).float()
    biases = torch.Tensor([-0.2723, 0.1896]).float()
    change_model_params(model, weights, biases)

    return model, DEVICE


def test_L1Regularization(model_config):
    model, dev_ = model_config
    regularizer = L1Regularization(regularization_rate=0.01, biases=False)

    obtained_term = regularizer(model.named_parameters(), device=dev_).item()
    expected_term = 0.022014999762177467

    msg = "Error in L1 regularization penalty"
    assert obtained_term == expected_term, msg


def test_L2Regularization(model_config):
    model, dev_ = model_config
    regularizer = L2Regularization(regularization_rate=0.01, biases=False)

    obtained_term = regularizer(model.named_parameters(), device=dev_).item()
    expected_term = 0.011890217661857605

    msg = "Error in L2 regularization penalty"
    assert obtained_term == expected_term, msg


def test_ElasticNetRegularization(model_config):
    # test checks if the linear combination is correct
    model, dev_ = model_config
    regularizer_l1 = L1Regularization(regularization_rate=0.01, biases=False)
    regularizer_l2 = L2Regularization(regularization_rate=0.01, biases=False)
    regularizer_elastic_l1 = ElasticNetRegularization(
        regularization_rate=0.01, alpha=1, biases=False
    )
    regularizer_elastic_l2 = ElasticNetRegularization(
        regularization_rate=0.01, alpha=0, biases=False
    )

    obtained_term_l1 = regularizer_l1(
        model.named_parameters(), device=dev_
    ).item()
    obtained_term_l2 = regularizer_l2(
        model.named_parameters(), device=dev_
    ).item()
    obtained_term_elastic_l1 = regularizer_elastic_l1(
        model.named_parameters(), device=dev_
    ).item()
    obtained_term_elastic_l2 = regularizer_elastic_l2(
        model.named_parameters(), device=dev_
    ).item()

    msg = "Error in ElasticNet L1"
    assert obtained_term_l1 == obtained_term_elastic_l1, msg

    msg = "Error in ElasticNet L2"
    assert obtained_term_l2 == obtained_term_elastic_l2, msg

    # ---------------------
    elastic = ElasticNetRegularization(
        regularization_rate=0.01, alpha=0.5, biases=False
    )
    obtained = elastic(model.named_parameters(), device=dev_).item()
    expected = 0.01695260778069496

    msg = "Error in ElasticNet"
    assert obtained == expected, msg
