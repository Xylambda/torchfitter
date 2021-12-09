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
    HISTORY : str
        Dictionary containing the metrics:
        * ParamsDict.HISTORY_TRAIN_LOSS
        * ParamsDict.HISTORY_VAL_LOSS
        * ParamsDict.HISTORY_LR
    HISTORY_TRAIN_LOSS : str
        Train loss for each epoch up to the current epoch.
    HISTORY_VAL_LOSS : str
        Validation loss for each epoch up to the current epoch.
    HISTORY_LR : str
        Learning rate for each epoch up to the current epoch.
    PROG_BAR : str
        Progress bar from tqdm library.
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
    HISTORY = "history"
    HISTORY_TRAIN_LOSS = "train_loss"
    HISTORY_VAL_LOSS = "validation_loss"
    HISTORY_LR = "learning_rate"
    PROG_BAR = 'progress_bar'
    TRAIN_LOADER = 'train_loader'
    TRAIN_BATCH = 'train_batch'
    TRAIN_BATCH_IDX = 'train_batch_idx'
    VAL_LOADER = 'validation_loader'
    VAL_BATCH = 'val_batch'
    VAL_BATCH_IDX = 'val_batch_idx'


class BarFormat:
    """
    Conventions for the bar formatting of tqdm progress bar. See references for
    more info.

    Parameters
    ----------
    FORMAT : str
        Bar format for tqdm progress bar.

    References
    ----------
    .. [1] tqdm - https://tqdm.github.io/docs/tqdm/#__init__
    """
    r_bar = '| {n_fmt}/{total_fmt} | {rate_noinv_fmt}{postfix}, time_remaining: {remaining} s'
    left = "{l_bar}{bar}"
    
    FORMAT = f"{left}{r_bar}"
