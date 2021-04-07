""" Regularization procedures for the Trainer class. """


from ._regularization_procedures import (
    L1Regularization,
    L2Regularization,
    ElasticNetRegularization,
)

# relative subpackages import
from . import base
from . import _regularization_procedures
