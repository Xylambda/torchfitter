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


@pytest.fixture
def loader_config():
    tensor_a = torch.Tensor([1, 2, 3, 4, 5, 6])
    tensor_b = torch.Tensor([7, 8, 9, 10, 11, 12])

    return tensor_a, tensor_b


def test_fast_tensor_dataloader_case1(loader_config):
    # len(tensors) % batch size == 0
    a, b = loader_config
    loader = FastTensorDataLoader(a, b, batch_size=3, shuffle=False)

    expected_dict = {}
    expected_dict['exp_a_iter0'] = torch.Tensor([1, 2, 3])
    expected_dict['exp_a_iter1'] = torch.Tensor([4, 5, 6])

    expected_dict['exp_b_iter0'] = torch.Tensor([7, 8, 9])
    expected_dict['exp_b_iter1'] = torch.Tensor([10, 11, 12])

    for i, (obt_a, obt_b) in enumerate(loader):
        key_a = f"exp_a_iter{i}"
        key_b = f"exp_b_iter{i}"

        torch.testing.assert_allclose(obt_a, expected_dict[key_a])
        torch.testing.assert_allclose(obt_b, expected_dict[key_b])


def test_fast_tensor_dataloader_case2(loader_config): 
    # len(tensors) % batch size != 0
    a, b = loader_config
    loader = FastTensorDataLoader(a, b, batch_size=5, shuffle=False)

    expected_dict = {}
    expected_dict['exp_a_iter0'] = torch.Tensor([1, 2, 3, 4, 5])
    expected_dict['exp_a_iter1'] = torch.Tensor([6])

    expected_dict['exp_b_iter0'] = torch.Tensor([7, 8, 9, 10, 11])
    expected_dict['exp_b_iter1'] = torch.Tensor([12])

    for i, (obt_a, obt_b) in enumerate(loader):
        key_a = f"exp_a_iter{i}"
        key_b = f"exp_b_iter{i}"

        torch.testing.assert_allclose(obt_a, expected_dict[key_a])
        torch.testing.assert_allclose(obt_b, expected_dict[key_b])