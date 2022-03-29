"""
Preprocessing functions.
"""
import math

import numpy as np
import torch
from sklearn.model_selection import train_test_split as __tr_test_split

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

    Examples
    --------
    >>> import torch
    >>> import numpy as np
    >>> from torchfitter.utils.preprocessing import numpy_to_torch
    >>> arr = np.array([1,2,3], dtype='int')
    >>> arr.dtype
    dtype('int64')

    >>> tensor = numpy_to_torch(arr, dtype='long')
    >>> tensor.dtype
    torch.int64
    """
    return getattr(torch.from_numpy(array), dtype)()


def train_test_val_split(
    X: np.ndarray,
    y: np.ndarray,
    train_ratio: float = 0.70,
    validation_ratio: float = 0.20,
    test_ratio: float = 0.10,
    random_state: int = 42,
    shuffle: bool = False,
    stratify=None,
):
    """
    Splits the given dataset into train, validation and test sets.

    This function applies `sklearn.model_selection.train_test_split` twice to
    generate the outputs.

    Parameters
    ----------
    X : array-like
        Features input set.
    y : array-like
        Labels input set.
    train_ratio : float, optional, default: 0.70
        Ratio of train set.
    validation_ratio : float, optional, default: 0.20
        Ratio of validation set.
    test_ratio : float, optional, default: 0.10
        Ratio of test set.
    random_state : int, optional, default: 42
        Controls the shuffling applied to the data before applying the split.
        Pass an int for reproducible output across multiple function calls.
    shuffle : bool, optional, default: False
        Whether or not to shuffle the data before splitting. If shuffle=False
        then stratify must be None. Shuffle will only be applied in the first
        split.
    stratify : array-like, default: None
        If not None, data is split in a stratified fashion, using this as the
        class labels. Stratify will only be applied in the first split.

    Returns
    -------
    X_train: np.ndarray
        Features train set.
    y_train: np.ndarray
        Labels train set.
    X_val: np.ndarray
        Features validation set.
    y_val: np.ndarray
        Labels validation set.
    X_test: np.ndarray
        Features test set.
    y_test : np.ndarray
        Labels test set.

    References
    ----------
    See `sklearn.model_selection.train_test_split`
    """
    msg = "'train_ratio', 'validation_ratio' and 'test_ratio' must sum 1"
    _sum = train_ratio + validation_ratio + test_ratio
    assert math.isclose(_sum, 1), msg

    test_size = 1 - train_ratio
    X_train, X_test, y_train, y_test = __tr_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        shuffle=shuffle,
        stratify=stratify,
    )

    val_size = test_ratio / (test_ratio + validation_ratio)
    X_val, X_test, y_val, y_test = __tr_test_split(
        X_test,
        y_test,
        test_size=val_size,
        random_state=random_state,
        shuffle=False,
        stratify=None,
    )

    return X_train, y_train, X_val, y_val, X_test, y_test
