""" Utils functions. """

import torch
import pickle
import numpy as np
from torch.utils.data import Dataset


class DataWrapper(Dataset):
    """
    Class to wrap a dataset. Assumes X and y are torch.Tensors or numpy.arrays.

    Parameters
    ----------
    X : torch.Tensor or numpy.array
        Features tensor.
    y : torch.Tensor or numpy.array
        Labels tensor.
    dtype_X : str, optional, default: 'float'
        Data type for features dataset.
    dtype_y : str, optional, default: 'int'
        Data type for labels dataset.
    """

    def __init__(self, X, y, dtype_X="float", dtype_y="int"):
        X, y = self._check_inputs(X, y, dtype_X, dtype_y)

        self.features = X
        self.labels = y

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

    def _check_inputs(self, X, y, dtype_X, dtype_y):
        """Check if the given inputs are numpy arrays and convert them if
        necessary.
        """
        if isinstance(X, np.ndarray):
            X = numpy_to_torch(X, dtype_X)

        if isinstance(y, np.ndarray):
            y = numpy_to_torch(y, dtype_y)

        return X, y


def numpy_to_torch(array, dtype):
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


def save_pickle(obj, path, protocol=pickle.HIGHEST_PROTOCOL):
    """
    Save given object as pickle.

    Parameters
    ----------
    obj : object
        Object to save.
    path : str or Path
        Path where to save the pickle.
    protocol : int, optional, default: pickle.HIGHEST_PROTOCOL
        Used pickle protocol.

    """
    with open(path, 'wb') as handle:
        pickle.dump(obj, handle, protocol=protocol)


def load_pickle(path):
    """
    Load saved pickle.

    Parameters
    ----------
    path : str or Path
        Path where to save the pickle.

    Returns
    -------
    obj : object
        Loaded object.

    """
    with open(path, 'rb') as handle:
        obj = pickle.load(handle)

    return obj
