""" PyTorch models fitting package. """

# relative subpackages import
from . import callbacks, conventions, io, testing, trainer, utils

__all__ = [
    "io",
    "utils",
    "trainer",
    "testing",
    "callbacks",
    "conventions",
]

from ._version import get_versions

__version__ = get_versions()["version"]
del get_versions
