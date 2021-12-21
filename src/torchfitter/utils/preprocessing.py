""" 
Preprocessing functions.
"""
import torch
import numpy as np


__all__ = ["numpy_to_torch", "train_test_val_split"]


def numpy_to_torch(array: np.ndarray, dtype: str) -> torch.Tensor:
    """
    Cast a numpy array to a torch tensor of the given dtype.

    Parameters
    ----------
    array : numpy.array
        Array to transform.
    dtype : str
        Desired data type.

    Returns
    -------
    tensor : torch.Tensor
        Torch tensor with the desired properties.
    """
    return getattr(torch.from_numpy(array), dtype)()


def train_test_val_split(X, y):
    """
    Splits the given dataset into train, validation and test sets.

    Parameters
    ----------
    X : array-like
    y : array-like

    Returns
    -------
    """
    raise NotImplementedError("func 'train_test_val_split' is not implemented")
