""" PyTorch models fitting package. """

# relative subpackages import
from . import utils
from . import trainer
from ._version import get_versions
__version__ = get_versions()['version']
del get_versions
