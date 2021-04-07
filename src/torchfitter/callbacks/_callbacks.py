""" """
import torch
import logging
from pathlib import Path
from .base import Callback
from torchfitter.conventions import ParamsDict
from torchfitter.utils import load_pickle, save_pickle


class EarlyStopping(Callback):
    """Callback to handle early stopping.

    `EarlyStopping` will be performed on the validation loss (as it should be).
    The best observed model will be loaded if `load_best` is True.

    Paramaters
    ----------
    patience : int, optional, default: 50
        Number of epochs to wait after min has been reached. After 'patience'
        number of epochs without improvemente, the training stops.
    load_best : bool, optional, deafult: True
        Whether to load the best observed parameters (True) or not (False).
    """

    def __init__(self, patience=50, load_best=True):
        super(EarlyStopping, self).__init__()
        self.patience = patience
        self.load_best = load_best

    def on_fit_start(self, params_dict):
        self.wait = 0
        self.stopped_epoch = 0
        self.best = float("inf")

    def on_epoch_end(self, params_dict):
        current_loss = params_dict[ParamsDict.VAL_LOS]
        epoch_number = params_dict[ParamsDict.EPOCH_NUMBER]
        model = params_dict[ParamsDict.MODEL]

        if current_loss < self.best:
            self.best = current_loss
            self.wait = 0
            # save weights
            self.best_params = model.state_dict()
        else:
            self.wait += 1

            if self.wait >= self.patience:
                self.stopped_epoch = epoch_number
                # send signal to stop training
                params_dict[ParamsDict.STOP_TRAINING] = True
                # load best weights
                if self.load_best:
                    model.load_state_dict(self.best_params)
                    logging.info("Best observed parameters loaded.")

    def on_fit_end(self, params_dict):
        if self.stopped_epoch > 0:
            logging.info(
                f"Early stopping applied at epoch: {self.stopped_epoch}"
            )


class LoggerCallback(Callback):
    """
    `LoggerCallback` is used to log some of the parameters dict in order to
    monitor the fitting process. The logged parameters are:
        - Current epoch.
        - Number of epochs.
        - Train loss.
        - Validation loss.
        - Time / epoch.

    It also outputs the total training time once the fitting process ends.

    Parameters
    ----------
    update_step : int, optional, default: 50
        Logs will be performed every 'update_step'.

    Attributes
    ----------
    header : list
        List of column names for the header.
    """

    def __init__(self, update_step=50):
        super(LoggerCallback, self).__init__()
        self.update_step = update_step

    def on_fit_start(self, params_dict):
        dev = params_dict[ParamsDict.DEVICE]
        logging.info(f"Starting training process on {dev}")

    def on_epoch_end(self, params_dict):
        # get params
        epochs = params_dict[ParamsDict.TOTAL_EPOCHS]
        epoch = params_dict[ParamsDict.EPOCH_NUMBER]
        val_loss = params_dict[ParamsDict.VAL_LOS]
        train_loss = params_dict[ParamsDict.TRAIN_LOSS]
        epoch_time = params_dict[ParamsDict.EPOCH_TIME]

        # log params
        if epoch % self.update_step == 0 or epoch == 1:
            tup = (epoch, epochs, train_loss, val_loss, round(epoch_time, 5))
            msg = (
                "Epoch: %-0i/%-8i | Train loss: %-13f | Validation loss: %-13f | Time/epoch: %-13f"
                % tup
            )
            logging.info(msg)

    def on_fit_end(self, params_dict):
        total_time = params_dict[ParamsDict.TOTAL_TIME]
        # final message
        logging.info(
            f"""End of training. Total time: {total_time:0.5f} seconds"""
        )


class LearningRateScheduler(Callback):
    """Callback to schedule learning rate.

    `LearningRateScheduler` provides an easy abstraction to schedule the
    learning rate of the optimizer by calling `scheduler.step()` after the
    training batch has been performed.

    Parameters
    ----------
    scheduler : torch.optim.lr_scheduler
        Torch learning rate scheduler.
    """

    def __init__(self, scheduler):
        super(LearningRateScheduler, self).__init__()
        self.scheduler = scheduler

    def on_train_batch_end(self, params_dict):
        self.scheduler.step()
