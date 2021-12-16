""" Internal conventions of torchfitter. """


class ParamsDict:
    """
    Naming conventions for torchfitter.trainer.Trainer internal parameters.

    Attributes
    ----------
    TRAIN_LOSS : str
        The current training loss.
    VAL_LOSS : str
        The current validation loss.
    OPTIMIZER : str
        Algorithm used to optimize the model.
    BATCH_TRAIN_LOSS : str
        Train loss of last computed training batch.
    BATCH_VAL_LOSS : str
        Validation loss of last computed validation batch.
    EPOCH_TIME : str
        The time it took to compute the current epoch.
    EPOCH_NUMBER : str
        The corresponding number of the current epoch.
    TOTAL_EPOCHS : str
        The total number of epochs.
    TOTAL_TIME : str
        The total time it took to complete all epochs.
    STOP_TRAINING : str
        The total time it took to complete all epochs.
    DEVICE : str
        Device where the model and data are stored.
    MODEL : str
        The model to train.
    ACCELERATOR : str
        The accelerate.Accelerator object used to boost the training process.
    BATCH_HISTORY : str
        History of the batch metrics. It is a dictionary
        containing the passed and the following metrics:
        * ParamsDict.HISTORY_TRAIN_LOSS
        * ParamsDict.HISTORY_VAL_LOSS
        * ParamsDict.HISTORY_LR
    EPOCH_HISTORY : str
        History of the epoch (accumulated batch) metrics. It is a dictionary
        containing the passed and the following metrics:
        * ParamsDict.HISTORY_TRAIN_LOSS
        * ParamsDict.HISTORY_VAL_LOSS
        * ParamsDict.HISTORY_LR
    LOSS : str
        Loss criterion historical value. It is a dictionary containing the keys
        'train' and 'validation'.
    HISTORY_LR : str
        Learning rate for each epoch up to the current epoch.
    TRAIN_LOADER : str
        Train dataloader.
    TRAIN_BATCH : str
        Current training batch.
    TRAIN_BATCH_IDX : str
        Current training batch index.
    VAL_LOADER : str
        Validation dataloader.
    VAL_BATCH : str
        Current validation batch.
    VAL_BATCH_IDX : str
        Current validation batch index.
    """
    TRAIN_LOSS = "training_loss"
    VAL_LOSS = "validation_loss"
    OPTIMIZER = 'optimizer'
    BATCH_TRAIN_LOSS = ""
    BATCH_VAL_LOSS = ""
    EPOCH_TIME = "epoch_time"
    EPOCH_NUMBER = "epoch_number"
    TOTAL_EPOCHS = "total_epochs"
    TOTAL_TIME = "total_time"
    STOP_TRAINING = "stop_training"
    DEVICE = "device"
    MODEL = "model"
    ACCELERATOR = "accelerator"
    BATCH_HISTORY = 'batch_history'
    EPOCH_HISTORY = 'epoch_history'
    LOSS = 'loss'
    HISTORY_LR = "learning_rate"
    TRAIN_LOADER = 'train_loader'
    TRAIN_BATCH = 'train_batch'
    TRAIN_BATCH_IDX = 'train_batch_idx'
    VAL_LOADER = 'validation_loader'
    VAL_BATCH = 'val_batch'
    VAL_BATCH_IDX = 'val_batch_idx'
