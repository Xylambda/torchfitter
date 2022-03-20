"""
Pool of utilities to wrap data.
"""
import torch
import numpy as np
from typing import Tuple, Union
from torch.utils.data import Dataset
from torchfitter.utils.preprocessing import numpy_to_torch


__all__ = [
    "DataWrapper",
    "FastTensorDataLoader",
]


class DataWrapper(Dataset):
    """
    Class to wrap a dataset. Assumes X and y are torch.Tensors or numpy.arrays.
    The DataWrapper will access the elements by indexing the first axis.

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

    def __init__(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        dtype_X: str = "float",
        dtype_y: str = "int",
    ):
        X, y = self._check_inputs(X, y, dtype_X, dtype_y)

        self.features = X
        self.labels = y

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.features[idx], self.labels[idx]

    def _check_inputs(
        self,
        X: Union[np.ndarray, torch.Tensor],
        y: Union[np.ndarray, torch.Tensor],
        dtype_X: float,
        dtype_y: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Check if the given inputs are numpy arrays and convert them if
        necessary.
        """
        if isinstance(X, np.ndarray):
            X = numpy_to_torch(X, dtype_X)
        else:
            X = getattr(X, dtype_X)()

        if isinstance(y, np.ndarray):
            y = numpy_to_torch(y, dtype_y)
        else:
            y = getattr(X, dtype_y)()

        return X, y


class FastTensorDataLoader:
    """DataLoader with faster loading.

    This class allows for a faster data loading. Although it won't always
    speed up the loading process, it can make the loading process 20 times
    faster.

    See `References` section to know the author.

    Parameters
    ----------
    *tensors : tuple of torch.Tensor
        Tensors to store.
    batch_size : int, optional, default: 32
        The batch size to load.
    shuffle : bool, optional, default: False
        Whether to shuffle the data (True) or not (False). If False, data will
        be processed in sequentially.

    References
    ----------
    .. [1] PyTorch discuss - Dataloader much slower than manual batching:
        https://discuss.pytorch.org/t/dataloader-much-slower-than-manual-
        batching/27014/6

    """

    def __init__(
        self,
        *tensors: torch.Tensor,
        batch_size: int = 32,
        shuffle: bool = False
    ):
        assert all(t.shape[0] == tensors[0].shape[0] for t in tensors)
        self.tensors = tensors

        self.dataset_len = self.tensors[0].shape[0]
        self.batch_size = batch_size
        self.shuffle = shuffle

        # Calculate # batches
        n_batches, remainder = divmod(self.dataset_len, self.batch_size)
        if remainder > 0:
            n_batches += 1

        self.n_batches = n_batches

    def __iter__(self):
        if self.shuffle:
            r = torch.randperm(self.dataset_len)
            self.tensors = [t[r] for t in self.tensors]
        self.i = 0

        return self

    def __next__(self) -> torch.Tensor:
        if self.i >= self.dataset_len:
            raise StopIteration

        start = self.i
        end = start + self.batch_size
        batch = tuple(t[start:end] for t in self.tensors)
        self.i += self.batch_size

        return batch

    def __len__(self) -> int:
        return self.n_batches
