""" Util functions for testing purposes. """
import torch
import torch.nn as nn


def change_model_params(model, weights, biases):
    """
    Change the model parameters. There is not need to return the model since
    the update will happen on the original object.

    Parameters
    ----------
    model : nn.Module
        Model to update.
    weights : torch.Tensor
        New weights.
    biases : torch.Tensor
        New biases.
    """
    with torch.no_grad():
        model.weight = nn.Parameter(weights)
        model.bias = nn.Parameter(biases)