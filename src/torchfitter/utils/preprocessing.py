"""
Preprocessing functions.
"""
import math
from typing import Iterable, List, Union

import numpy as np
import torch
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split as __tr_test_split

__all__ = [
    "numpy_to_torch",
    "train_test_val_split",
    "torch_to_numpy",
    "tabular_to_sliding_dataset",
]


def torch_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """
    Cast a torch.Tensor to a numpy.ndarray dealing with device management if
    any. For example, a tensor may need to be detached but it is not stored on
    the cpu.

    Parameters
    ----------
    tensor : torch.Tensor
        Tensor to convert to numpy.

    Returns
    -------
    array : numpy.array
        NumPy array.
    """
    try:
        array = tensor.detach().numpy()
    except Exception:
        array = tensor.cpu().detach().numpy()

    return array


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
) -> Iterable[np.ndarray]:
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
    X_train: numpy.ndarray
        Features train set.
    y_train: numpy.ndarray
        Labels train set.
    X_val: numpy.ndarray
        Features validation set.
    y_val: numpy.ndarray
        Labels validation set.
    X_test: numpy.ndarray
        Features test set.
    y_test : numpy.ndarray
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


def tabular_to_sliding_dataset(
    dataset: np.ndarray,
    validation_idx: int,
    test_idx: int,
    n_past: int,
    n_future: int,
    make_writable: bool = True,
    scaler: Union[TransformerMixin, BaseEstimator] = None,
) -> List[np.ndarray]:
    """Convert a tabular or 2D dataset to a sliding window dataset (3D).

    This function expects a datatype that supports the array protocol.
    E.g.: Pandas DataFrame or NumPy arrays.

    Parameters
    ----------
    dataset : array-like
        Array-like object.
    validation_idx : int
        Index to create the validation set.
    test_idx : int
        Index to create the testing set.
    n_past : int
        Number of past steps to make predictions. It will be used to
        generate the features.
    n_future : int
        Number of future steps to predict. It will be used to generate
        the labels.
    make_writable : bool, optional, default: True
        Make the resulting arrays writable by creating a copy of the view.
    scaler : sklearn.base.TransformerMixin, optional, default: None
        If not None, the data will be normalized with the passed scaler.
        Assumes distribution does not vary over time.

    Returns
    -------
    output : list of numpy.ndarray
        A list containing the resulting arrays. They appear in this order:
            * X_train: Features train set.
            * y_train: Labels train set.
            * X_val: Features validation set.
            * y_val: Labels validation set.
            * X_test: Features test set.
            * y_test : Labels test set.

    Warning
    -------
    This function is very memory-consuming.

    See Also
    --------
    torchfitter.utils.preprocessing.train_test_val_split

    TODO
    ----
    * Allow spliting by percentage.
    * Allow single-feature forecasting instead of multi-forecasting.
    * Use `train_test_val_split` to abstract the splitting.
    * Allow selecting the target column.
    """

    def get_train_and_test(array, n_past, n_future):
        """
        Convenient sub-function that wraps to functionality to
        create a rolling view and select the past as features
        and the future as labels.
        """
        window_length = n_past + n_future
        roll_view = np.lib.stride_tricks.sliding_window_view(
            array, window_length, axis=0
        )
        X = roll_view[:, :, :n_past]
        y = roll_view[:, :, n_past:]
        return X, y

    # type-agnostic
    arr = dataset.__array__()
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)

    # split
    train = arr[:validation_idx]
    validation = arr[validation_idx:test_idx]
    test = arr[test_idx:]

    if scaler is not None:
        scaler.fit(train)

        train = scaler.transform(train)
        validation = scaler.transform(validation)
        test = scaler.transform(test)

    # get a rolling view of each data chunk
    output = []
    for chunk in [train, validation, test]:
        X, y = get_train_and_test(
            array=chunk, n_past=n_past, n_future=n_future
        )

        # make a copy to generate a writable array
        if make_writable:
            _tup = (X.copy(), y.copy())
        else:
            _tup = (X, y)

        output.append(_tup)

    # unpack and return
    output = [item for sublist in output for item in sublist]
    return output
