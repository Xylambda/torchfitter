""" PyTorch models fitting package. """

# relative subpackages import
from . import io
from . import utils
from . import trainer
from . import testing
from . import manager
from . import callbacks
from . import conventions
from . import regularization

__all__ = [
    "io",
    "utils",
    "trainer",
    "testing",
    "manager",
    "callbacks",
    "conventions",
    "regularization",
]

from ._version import get_versions

__version__ = get_versions()["version"]
del get_versions
