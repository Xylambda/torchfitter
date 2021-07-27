""" Abstractions to interact with models. """

from . import base
from ._trainer_callbacks import (
    EarlyStopping,
    LoggerCallback,
    LearningRateScheduler,
    ReduceLROnPlateau,
    GPUStats
)
