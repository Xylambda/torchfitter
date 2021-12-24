import torch
import pytest
import numpy as np
from torchfitter.utils.preprocessing import (
    numpy_to_torch, train_test_val_split
)
from torchfitter.utils.data import DataWrapper, FastTensorDataLoader


def test_datawrapper():
    X = np.array([1, 2, 3, 4, 5])
    y = np.array([1, 2, 3, 4, 5])

    wrapper = DataWrapper(X, y, dtype_X="int", dtype_y="int")

    X_expected = torch.Tensor([1, 2, 3, 4, 5]).int()
    y_expected = torch.Tensor([1, 2, 3, 4, 5]).int()

    torch.testing.assert_allclose(wrapper.features, X_expected)
    torch.testing.assert_allclose(wrapper.labels, y_expected)


def test_numpy_to_torch():
    arr = np.random.rand(10)
    tensor = numpy_to_torch(arr, "float")

    msg = f"Numpy array of type '{type(arr)}' not casted to torch.tensor"
    assert isinstance(tensor, torch.Tensor), msg

    msg = f"Torch tensor should be 'torch.float32' but '{tensor.dtype}' found"
    assert tensor.dtype == torch.float32, msg


def test_train_test_val_split():
    X = np.array([x for x in range(10)])
    y = np.array([y for y in range(10,20)])

    X_train, y_train, X_val, y_val, X_test, y_test =  train_test_val_split(
        X, y, shuffle=False
    )

    X_train_expected = np.array([0, 1, 2, 3, 4, 5])
    y_train_expected = np.array([10, 11, 12, 13, 14, 15])

    X_val_expected = np.array([6, 7])
    y_val_expected = np.array([16, 17])

    X_test_expected = np.array([8, 9])
    y_test_expected = np.array([16, 17])

    np.testing.assert_allclose(X_train, X_train_expected)
    np.testing.assert_allclose(y_train, y_train_expected)

    np.testing.assert_allclose(X_val, X_val_expected)
    np.testing.assert_allclose(y_val, y_val_expected)

    np.testing.assert_allclose(X_test, X_test_expected)
    np.testing.assert_allclose(y_val, y_test_expected)


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
    expected_dict["exp_a_iter0"] = torch.Tensor([1, 2, 3])
    expected_dict["exp_a_iter1"] = torch.Tensor([4, 5, 6])

    expected_dict["exp_b_iter0"] = torch.Tensor([7, 8, 9])
    expected_dict["exp_b_iter1"] = torch.Tensor([10, 11, 12])

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
    expected_dict["exp_a_iter0"] = torch.Tensor([1, 2, 3, 4, 5])
    expected_dict["exp_a_iter1"] = torch.Tensor([6])

    expected_dict["exp_b_iter0"] = torch.Tensor([7, 8, 9, 10, 11])
    expected_dict["exp_b_iter1"] = torch.Tensor([12])

    for i, (obt_a, obt_b) in enumerate(loader):
        key_a = f"exp_a_iter{i}"
        key_b = f"exp_b_iter{i}"

        torch.testing.assert_allclose(obt_a, expected_dict[key_a])
        torch.testing.assert_allclose(obt_b, expected_dict[key_b])
