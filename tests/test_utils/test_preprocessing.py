import torch
import pytest
import numpy as np
from torchfitter.utils.preprocessing import (
    numpy_to_torch,
    train_test_val_split,
    torch_to_numpy,
    tabular_to_sliding_dataset,
)


def test_numpy_to_torch():
    arr = np.random.rand(10)
    tensor = numpy_to_torch(arr, "float")

    msg = f"Numpy array of type '{type(arr)}' not casted to torch.tensor"
    assert isinstance(tensor, torch.Tensor), msg

    msg = f"Torch tensor should be 'torch.float32' but '{tensor.dtype}' found"
    assert tensor.dtype == torch.float32, msg


@pytest.mark.xfail(reason="Need to be tested with GPUs")
def test_torch_to_numpy():
    pass


def test_tabular_to_sliding_dataset():
    dataset = np.arange(30)

    # -------------------------------------------------------------------------
    # expected train
    X_train_expected = np.array(
        [
            [[0, 1, 2]],
            [[1, 2, 3]],
            [[2, 3, 4]],
            [[3, 4, 5]],
            [[4, 5, 6]],
            [[5, 6, 7]],
            [[6, 7, 8]],
            [[7, 8, 9]],
            [[8, 9, 10]],
            [[9, 10, 11]],
            [[10, 11, 12]],
            [[11, 12, 13]],
            [[12, 13, 14]],
            [[13, 14, 15]],
            [[14, 15, 16]],
            [[15, 16, 17]],
            [[16, 17, 18]],
        ]
    )

    y_train_expected = np.array(
        [
            [[3]],
            [[4]],
            [[5]],
            [[6]],
            [[7]],
            [[8]],
            [[9]],
            [[10]],
            [[11]],
            [[12]],
            [[13]],
            [[14]],
            [[15]],
            [[16]],
            [[17]],
            [[18]],
            [[19]],
        ]
    )

    # expected validation
    X_val_expected = np.array([[[20, 21, 22]], [[21, 22, 23]]])
    y_val_expected = np.array([[[23]], [[24]]])

    # expected test
    X_test_expected = np.array([[[25, 26, 27]], [[26, 27, 28]]])
    y_test_expected = np.array([[[28]], [[29]]])

    # -------------------------------------------------------------------------
    obtained = tabular_to_sliding_dataset(
        dataset=dataset, validation_idx=20, test_idx=25, n_past=3, n_future=1
    )
    X_train, y_train, X_val, y_val, X_test, y_test = obtained

    # -------------------------------------------------------------------------
    np.testing.assert_almost_equal(X_train_expected, X_train)
    np.testing.assert_almost_equal(y_train_expected, y_train)

    np.testing.assert_almost_equal(X_val_expected, X_val)
    np.testing.assert_almost_equal(y_val_expected, y_val)

    np.testing.assert_almost_equal(X_test_expected, X_test)
    np.testing.assert_almost_equal(y_test_expected, y_test)


def test_train_test_val_split():
    X = np.array([x for x in range(10)])
    y = np.array([y for y in range(10, 20)])

    X_train, y_train, X_val, y_val, X_test, y_test = train_test_val_split(
        X, y, shuffle=False
    )

    X_train_expected = np.array([0, 1, 2, 3, 4, 5])
    y_train_expected = np.array([10, 11, 12, 13, 14, 15])

    X_val_expected = np.array([6, 7])
    y_val_expected = np.array([16, 17])

    X_test_expected = np.array([8, 9])
    y_test_expected = np.array([18, 19])

    np.testing.assert_allclose(X_train, X_train_expected)
    np.testing.assert_allclose(y_train, y_train_expected)

    np.testing.assert_allclose(X_val, X_val_expected)
    np.testing.assert_allclose(y_val, y_val_expected)

    np.testing.assert_allclose(X_test, X_test_expected)
    np.testing.assert_allclose(y_test, y_test_expected)
