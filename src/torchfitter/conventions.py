""" Internal conventions of torchfitter. """


class ParamsDict:
    TRAIN_LOSS = "training_loss"
    VAL_LOS = "validation_loss"
    EPOCH_TIME = "epoch_time"
    EPOCH_NUMBER = "epoch_number"
    TOTAL_EPOCHS = "total_epochs"
    TOTAL_TIME = "total_time"
    STOP_TRAINING = "stop_training"
    DEVICE = "device"
    MODEL = "model"
    HISTORY = "history"
    HISTORY_TRAIN_LOSS = "train_loss"
    HISTORY_VAL_LOSS = "validation_loss"
    HISTORY_LR = "learning_rate"
