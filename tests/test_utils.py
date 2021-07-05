import torch
import pytest
import numpy as np
from torchfitter.utils import DataWrapper, numpy_to_torch, FastTensorDataLoader


def test_datawrapper():
    X = np.array([1,2,3,4,5])
    y = np.array([1,2,3,4,5])

    wrapper = DataWrapper(X, y, dtype_X='int', dtype_y='int')

    X_expected = torch.Tensor([1,2,3,4,5]).int()
    y_expected = torch.Tensor([1,2,3,4,5]).int()

    torch.testing.assert_allclose(wrapper.features, X_expected)
    torch.testing.assert_allclose(wrapper.labels, y_expected)


def test_numpy_to_torch():
    arr = np.random.rand(10)
    tensor = numpy_to_torch(arr, 'float')

    msg = f"Numpy array of type '{type(arr)}' not casted to torch.tensor"
    assert isinstance(tensor, torch.Tensor), msg

    msg = f"Torch tensor should be 'torch.float32' but '{tensor.dtype}' found"
    assert tensor.dtype == torch.float32, msg


@pytest.mark.xfail
def test_fast_tensor_dataloader():
    pass