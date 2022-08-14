""" Abstractions to interact with models. """

from torchfitter.callbacks.progress import LoggerCallback, RichProgressBar
from torchfitter.callbacks.regularization import (
    ElasticNetRegularization,
    L1Regularization,
    L2Regularization,
)

from . import base
from ._callbacks import (
    EarlyStopping,
    GPUStats,
    LearningRateScheduler,
    StochasticWeightAveraging,
)

__all__ = [
    "base",
    "GPUStats",
    "EarlyStopping",
    "LoggerCallback",
    "RichProgressBar",
    "LearningRateScheduler",
    "StochasticWeightAveraging",
    "L1Regularization",
    "L2Regularization",
    "ElasticNetRegularization",
]
