""" """
import torch
import logging
from pathlib import Path
from .base import Callback


class EarlyStopping(Callback):
    """
    Callback to handle early stopping.

    Paramaters
    ----------
    patience : int, optional, default: 50
        Number of epochs to wait after min has been reached. After 'patience'
        number of epochs without improvemente, the training stops.
    """
    def __init__(self, patience=50, path=None):
        super(EarlyStopping, self).__init__()
        self.patience = patience

    def _save_params(self, parameters):
        if self.path is not None:
            torch.save(parameters, self.path / 'model_params.pth')

    def on_fit_start(self, params_dict):
        self.wait = 0
        self.stopped_epoch = 0
        self.best = float('inf')

    def on_epoch_end(self, params_dict):
        current_loss = params_dict['validation_loss']
        epoch_number = params_dict['epoch_number']
        model = params_dict['model']

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
                params_dict['model'].stop_training = True
                # load best weights
                model.load_state_dict(self.best_params)

    def on_fit_end(self, params_dict):
        if self.stopped_epoch > 0:
            logging.info(f"--- Early stopped at epoch: {self.stopped_epoch} ---")


class LoggerCallback(Callback):
    """
    Callback to log basic data.

    Parameters
    ----------
    update_step : int, optional, default: 50
        Logs will be performed every 'update_step'.
    """
    def __init__(self, update_step=50):
        super(LoggerCallback, self).__init__()
        self.update_step = update_step

    def on_fit_start(self, params_dict):
        dev = params_dict['device']
        logging.info(f"Starting training process on {dev}")

    def on_epoch_end(self, params_dict):
        # get params
        epochs = params_dict['total_epochs']
        epoch = params_dict['epoch_number']
        val_loss = params_dict['validation_loss']
        train_loss = params_dict['training_loss']
        epoch_time = params_dict['epoch_time']

        # log params
        if epoch % self.update_step == 0 or epoch == 1:
            """
            msg = f"Epoch {epoch}/{epochs} | Train loss: {train_loss}"
            msg = f"{msg} | Validation loss: {val_loss}"
            msg = f"{msg} | Time/epoch: {round(epoch_time, 5)} seconds"
            """
            msg = f"Epoch {epoch}/{epochs:16} | Train loss: {train_loss:16} | Validation loss: {val_loss:16} | Time/epoch: {round(epoch_time, 5):16} seconds"
            logging.info(msg)

    def on_fit_end(self, params_dict):
        total_time = params_dict['total_time']
        # final message
        logging.info(
            f"""End of training. Total time: {round(total_time, 5)} seconds"""
        )