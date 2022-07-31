""" Pool of miscellaneous and convenient functions. """

import logging

import torch

__all__ = [
    "check_model_on_cuda",
    "get_logger",
]


def check_model_on_cuda(model: torch.nn.Module) -> bool:
    """
    Check if the model is stored in a cuda device.

    Parameters
    ----------
    model : torch.nn.Module
        PyTorch model to check.

    Return
    ------
    bool
        True if the model is stored on a cuda device.
    """
    return next(model.parameters()).is_cuda


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Generate a logger with the specified name and level.

    Parameters
    ----------
    name : str
        Logger name.
    level : int
        Logging level for the logger.

    Returns
    -------
    logger : logging.Logger
        Logger.
    """
    logger = logging.getLogger(name=name)
    logger.setLevel(level=level)
    return logger


def freeze_model(model: torch.nn.Module) -> None:
    """Freeze the given model.

    This function is an inplace operations that deactivates the gradient of all
    parameters.

    Parameters
    ----------
    models : torch.nn.Module
        Model to freeze.
    """
    for param in model.parameters():
        param.requires_grad = False


def unfreeze_model(model: torch.nn.Module) -> None:
    """Unfreeze the given model.

    This function is an inplace operations that activates the gradient of all
    parameters.

    Parameters
    ----------
    models : torch.nn.Module
        Model to unfreeze.
    """
    for param in model.parameters():
        param.requires_grad = True
