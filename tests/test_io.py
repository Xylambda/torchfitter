import os
import pytest
from torchfitter import io
from pathlib import Path


_path = Path(__file__).parent
DATA_PATH = _path / "data"


def test_save_pickle():
    a = [1, 2, 3]
    io.save_pickle(obj=a, path=DATA_PATH / 'saving_test.pkl')
    assert 'saving_test.pkl' in os.listdir(DATA_PATH), "Pickle not saved appropiately"
    os.remove(DATA_PATH / 'saving_test.pkl')


def test_load_pickle():
    expected = [1, 2, 3]
    obtained = io.load_pickle(DATA_PATH / 'pickle_test.pkl')
    assert expected == obtained, "Pickle not loaded correctly"