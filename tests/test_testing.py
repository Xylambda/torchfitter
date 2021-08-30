import torch
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


def test_compute_forward_gradient():
    class TestModule(torch.nn.Module):
        def __init__(self):
            super(TestModule, self).__init__()
            
        def forward(self, a, b):
            return (a - b).mean()

    test = TestModule()
    a = torch.ones(2, requires_grad=True)
    b = torch.zeros(2, requires_grad=True)

    obtained = compute_forward_gradient(test, a, b)
    expected = {0: torch.Tensor([0.5, 0.5]), 1: torch.Tensor([-0.5, -0.5])}

    for k1, k2 in zip(obtained, expected):
        torch.testing.assert_allclose(obtained[k1], expected[k2])


def test_check_monotonically_decreasing():
    iterable_1 = [5, 4, 3, 3]
    iterable_2 = [5, 4, 3, 2]

    msg = f"'check_monotonically_decreasing' should be True for {iterable_1}"
    assert check_monotonically_decreasing(iterable_1, strict=False), msg

    msg = f"'check_monotonically_decreasing' should be True for {iterable_2}"
    assert check_monotonically_decreasing(iterable_2, strict=False), msg


def test_check_monotonically_decreasing_strict():
    iterable_1 = [5, 4, 3, 3]
    iterable_2 = [5, 4, 3, 2]

    msg = f"'check_monotonically_decreasing' should be False for {iterable_1}"
    assert not check_monotonically_decreasing(iterable_1, strict=True), msg

    msg = f"'check_monotonically_decreasing' should be True for {iterable_2}"
    assert check_monotonically_decreasing(iterable_2, strict=True), msg