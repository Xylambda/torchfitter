""" Util functions for testing purposes. """
from typing import Iterable

import torch
import torch.nn as nn


def change_model_params(
    model: torch.nn.Module, weights: torch.Tensor, biases: torch.Tensor
) -> None:
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


def compute_forward_gradient(module: torch.nn.Module, *tensors) -> dict:
    """Computes forward gradient.

    This function helps to test the gradient computation of a nn.Module class.

    It computes the gradients for the given variables. The result of a forward
    must be a torch.Tensor with the shape (1,1), otherwise is not possible to
    compute the gradients.

    Parameters
    ----------
    module : torch.nn.Module
        The module function with a defined forward.
    *tensors : tuple
        The input tensors to compute the gradients of.

    Returns
    -------
    gradients : dict
        A dictionary whose keys are a list of sorted integers and whose values
        are the gradients for each passed tensor (in the same order they were
        passed).

    Note
    ----
    This function returns the gradients for the leaf variables, not the
    intermediate gradients.

    """
    # minimal check
    for t in tensors:
        if not t.requires_grad:
            raise ValueError("Tensors must have 'requires_grad' activated.")

    # create computational graph
    forward = module(*tensors)

    # compute gradients
    if forward.shape != torch.Size([1]) and forward.shape != torch.Size([]):
        raise ValueError(
            "The passed tensors produced a vector result instead of a scalar "
            "result. It is not possible to compute the backward pass if the "
            "forward result is not a (1,1) torch.Tensor."
        )
    else:
        forward.backward()

    # store gradient for each passed tensor
    gradients = {}
    for i, t in enumerate(tensors):
        gradients[i] = t.grad

    return gradients


def check_monotonically_decreasing(
    iterable: Iterable[float], strict: bool = False
) -> bool:
    """Check if the given iterable is monotonically decreasing.

    The function allows to check strictly (all i + 1 are greater than i) or
    non-strictly (all i + 1 are greater or equal than i).

    Parameters
    ----------
    iterable : array-like
        Iterable object to check.
    strict : bool, optional, default: False
        Whether to strictly check monotonically decreasing (True) or not
        (False).

    Returns
    -------
    bool
        True if 'iterable' is monotonically decreasing.

    References
    ----------
    .. [1] Python - How to check list monotonicity
       https://stackoverflow.com/questions/4983258/python-how-to-check-list-
       monotonicity
    """
    _zip = zip(iterable, iterable[1:])
    if strict:
        return all(next_ > current for next_, current in _zip)
    else:
        return all(next_ >= current for next_, current in _zip)
