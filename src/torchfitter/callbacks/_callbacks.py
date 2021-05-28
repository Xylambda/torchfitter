""" """
import torch
import logging
from .base import Callback
from torchfitter.conventions import ParamsDict


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
    path : str or Path, optional, default: 'checkpoint'
        Path to store the best observed parameters.
    """

    def __init__(self, patience=50, load_best=True, path='checkpoint.pt'):
        super(EarlyStopping, self).__init__()
        self.path = path
        self.patience = patience
        self.load_best = load_best

        # to restart callback state
        self.__restart_dict = {
            'patience': patience,
            'load_best': load_best,
            'path': path # not really necessary
        }

    def __repr__(self) -> str:
        return f"EarlyStopping(patience={self.patience}, load_best={self.load_best})"

    def on_fit_start(self, params_dict):
        self.wait = 0
        self.stopped_epoch = 0
        self.best = float("inf")

        # update __restart_dict
        self.__restart_dict['wait'] = 0
        self.__restart_dict['stopped_epoch'] = 0
        self.__restart_dict['best'] = float("inf")

    def on_epoch_end(self, params_dict):
        current_loss = params_dict[ParamsDict.VAL_LOSS]
        epoch_number = params_dict[ParamsDict.EPOCH_NUMBER]
        model = params_dict[ParamsDict.MODEL]

        if current_loss < self.best:
            self.best = current_loss
            self.wait = 0
            # save weights
            best_params = model.state_dict().copy()
            torch.save(best_params, self.path)
        else:
            self.wait += 1

            if self.wait >= self.patience:
                self.stopped_epoch = epoch_number
                # send signal to stop training
                params_dict[ParamsDict.STOP_TRAINING] = True
                # load best weights
                if self.load_best:
                    best_params = torch.load(self.path)
                    model.load_state_dict(best_params)
                    logging.info("Best observed parameters loaded.")

    def on_fit_end(self, params_dict):
        if self.stopped_epoch > 0:
            logging.info(
                f"Early stopping applied at epoch: {self.stopped_epoch}"
            )

    def reset_parameters(self):
        for key, value in self.__restart_dict.items():
            self.__dict__[key] = value


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

    def __repr__(self) -> str:
        return f"LoggerCallback(update_step={self.update_step})"

    def on_fit_start(self, params_dict):
        dev = params_dict[ParamsDict.DEVICE]
        logging.info(f"Starting training process on {dev}")

    def on_epoch_end(self, params_dict):
        # get params
        epochs = params_dict[ParamsDict.TOTAL_EPOCHS]
        epoch = params_dict[ParamsDict.EPOCH_NUMBER]
        val_loss = params_dict[ParamsDict.VAL_LOSS]
        train_loss = params_dict[ParamsDict.TRAIN_LOSS]
        epoch_time = params_dict[ParamsDict.EPOCH_TIME]
        pbar = params_dict[ParamsDict.PROG_BAR]

        pbar.set_description(f"Epoch: {epoch}/{epochs}")
        pbar.set_postfix(
            train_loss=train_loss,
            val_loss=val_loss,
            epoch_time=f"{epoch_time:.2f} s",
            refresh=True
        )

    def on_fit_end(self, params_dict):
        total_time = params_dict[ParamsDict.TOTAL_TIME]
        # final message
        logging.info(
            f"""End of training. Total time: {total_time:0.5f} seconds"""
        )

    def reset_parameters(self):
        pass # no need to restart any parameter


class LearningRateScheduler(Callback):
    """Callback to schedule learning rate.

    `LearningRateScheduler` provides an easy abstraction to schedule the
    learning rate of the optimizer by calling `scheduler.step()` after the
    training batch has been performed.

    Parameters
    ----------
    scheduler : torch.optim.lr_scheduler
        Torch learning rate scheduler.
    metric : str, optional, default: None
        Metric to read. Needed for some optimizers. See 
        torchfitter.convetions.ParamsDict for available metrics.

    References
    ----------
    ..[1] PyTorch - How to adjust learning rate
       https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate
    """

    def __init__(self, scheduler, metric=None):
        super(LearningRateScheduler, self).__init__()
        self.scheduler = scheduler
        self.metric = metric

        # to restart callback state
        self.__restart_dict = {
            'metric': metric,
            'scheduler': scheduler,
            'scheduler_params': scheduler.state_dict().copy()
        }

    def __repr__(self) -> str:
        return f"LearningRateScheduler(scheduler={self.scheduler}, metric={self.metric})"

    def on_train_batch_end(self, params_dict):
        if params_dict[ParamsDict.EPOCH_NUMBER] == 1:
            pass
        else:
            if self.metric is None:
                self.scheduler.step()
            else:
                metric = params_dict[self.metric]
                self.scheduler.step(metric)

    def reset_parameters(self) -> None:
        for key, value in self.__restart_dict.items():
            if key == 'scheduler_params':
                self.__dict__['scheduler'].load_state_dict(value)
            else:
                self.__dict__[key] = value