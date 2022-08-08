import torch
from torchfitter.utils.convenience import freeze_model, unfreeze_model


def test_freeze_model():
    model = torch.nn.Linear(3, 3)
    freeze_model(model)

    msg = "Parameter not being freezed"
    for param in model.parameters():
        assert param.requires_grad is False, msg


def test_unfreeze_model():
    model = torch.nn.Linear(3, 3)

    # explicitly freeze
    for param in model.parameters():
        param.requires_grad = False

    unfreeze_model(model)

    msg = "Parameter not being unfreezed"
    for param in model.parameters():
        assert param.requires_grad is True, msg
