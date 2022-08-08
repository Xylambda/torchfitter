""" PyTorch models fitting package. """

# relative subpackages import
from . import (
    callbacks,
    conventions,
    io,
    manager,
    testing,
    trainer,
    utils,
)

__all__ = [
    "io",
    "utils",
    "trainer",
    "testing",
    "manager",
    "callbacks",
    "conventions",
]

from ._version import get_versions

__version__ = get_versions()["version"]
del get_versions
