""" Internal conventions of torchfitter. """


class ParamsDict:
    TRAIN_LOSS = "training_loss"
    VAL_LOS = "validation_loss"
    TRAIN_BATCH = "train_batch"
    VAL_BATCH = "validation_batch"
    EPOCH_TIME = "epoch_time"
    HISTORY = "history"
    HISTORY_TRAIN_LOSS = "history_training"
    HISTORY_VAL_LOSS = "history_validation"
