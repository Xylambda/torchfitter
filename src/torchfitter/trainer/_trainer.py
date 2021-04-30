import time
import torch
import pickle
import logging
import warnings
import statistics

from tqdm.auto import tqdm
from torchfitter.conventions import ParamsDict
from torchfitter.callbacks.base import CallbackHandler
from torchfitter.utils import load_pickle, save_pickle


class Trainer:
    """Trainer

    Class that eases the training of a PyTorch model.

    Parameters
    ----------
    model : torch.Module
        The model to train.
    criterion : torch.Module
        Loss function criterion.
    optimizer : torch.optim
        Optimizer to perform the parameters update.
    regularizer : torchfitter.regularizer, optional, default: None
        Procedure to apply penalties to the loss function.
    device : str, optional, default: None
        Device to perform computations. If None, the Trainer will automatically
        select the device.
    callbacks : list of torchfitter.callback.Callback
        Callbacks that allow interaction.

    Attributes
    ----------
    callback_handler : torchfitter.callback.CallbackHandler
        Handles the passed callbacks.
    params_dict : dict
        Contains training params.

    """

    def __init__(
        self,
        model,
        criterion,
        optimizer,
        regularizer=None,
        device=None,
        callbacks=None,
    ):
        self.criterion = criterion
        self.optimizer = optimizer
        self.regularizer = regularizer
        self.device = self._get_device(device)
        self.model = model.to(self.device) # DISC: is it really necessary?

        # attributes
        self.callback_handler = CallbackHandler(callbacks_list=callbacks)
        self.params_dict = self._initialize_params_dict()

        logging.basicConfig(level=logging.INFO)

    def fit(self, train_loader, val_loader, epochs):
        """Fits.

        Fit the model using the given loaders for the given number of epochs.

        Parameters
        ----------
        train_loader : torch.DataLoader
            DataLoader containing train dataset.
        val_loader : torch.DataLoader
            DataLoader containing validation dataset.
        epochs : int
            Number of training epochs.

        """
        # track total training time
        total_start_time = time.time()

        self.callback_handler.on_fit_start(self.params_dict)
        initial_epoch = self.params_dict[ParamsDict.EPOCH_NUMBER]

        # ---- train process ----
        for epoch in tqdm(range(initial_epoch, epochs+1), ascii=True):
            self.callback_handler.on_epoch_start(self.params_dict)

            # track epoch time
            epoch_start_time = time.time()

            # train
            self.callback_handler.on_train_batch_start(self.params_dict)
            tr_loss = self._train(train_loader)
            self.callback_handler.on_train_batch_end(self.params_dict)

            # validation
            self.callback_handler.on_validation_batch_start(self.params_dict)
            val_loss = self._validate(val_loader)
            self.callback_handler.on_validation_batch_end(self.params_dict)

            self._update_history(
                **{
                    ParamsDict.HISTORY_TRAIN_LOSS: tr_loss,
                    ParamsDict.HISTORY_VAL_LOSS: val_loss,
                    ParamsDict.HISTORY_LR: self.optimizer.param_groups[0]["lr"],
                }
            )

            epoch_time = time.time() - epoch_start_time
            self._update_params_dict(
                **{
                    ParamsDict.VAL_LOS: val_loss,
                    ParamsDict.TRAIN_LOSS: tr_loss,
                    ParamsDict.EPOCH_TIME: epoch_time,
                    ParamsDict.EPOCH_NUMBER: epoch,
                    ParamsDict.TOTAL_EPOCHS: epochs,
                }
            )

            self.callback_handler.on_epoch_end(self.params_dict)

            if self.params_dict[ParamsDict.STOP_TRAINING]:
                break

        total_time = time.time() - total_start_time

        self._update_params_dict(**{ParamsDict.TOTAL_TIME: total_time})
        self.callback_handler.on_fit_end(self.params_dict)

    def _update_params_dict(self, **kwargs):
        """
        Update paramaters dictionary with the passed key-value pairs.

        Parameters
        ----------
        kwargs : dict
            Dictionary with keys to update.
        """
        for key, value in kwargs.items():
            self.params_dict[key] = value

    def _update_history(self, train_loss, validation_loss, learning_rate):
        self.params_dict[ParamsDict.HISTORY][
            ParamsDict.HISTORY_TRAIN_LOSS
        ].append(train_loss)
        self.params_dict[ParamsDict.HISTORY][
            ParamsDict.HISTORY_VAL_LOSS
        ].append(validation_loss)
        self.params_dict[ParamsDict.HISTORY][ParamsDict.HISTORY_LR].append(
            learning_rate
        )

    def _initialize_params_dict(self):
        params_dict = {
            ParamsDict.TRAIN_LOSS: float('inf'),
            ParamsDict.VAL_LOS: float('inf'),
            ParamsDict.EPOCH_TIME: 0,
            ParamsDict.EPOCH_NUMBER: 1,
            ParamsDict.TOTAL_EPOCHS: None,
            ParamsDict.TOTAL_TIME: 0,
            ParamsDict.STOP_TRAINING: False,
            ParamsDict.DEVICE: self.device,
            ParamsDict.MODEL: self.model,
            ParamsDict.HISTORY: {
                ParamsDict.HISTORY_TRAIN_LOSS: [],
                ParamsDict.HISTORY_VAL_LOSS: [],
                ParamsDict.HISTORY_LR: [],
            },
        }

        return params_dict

    def reset_parameters(self):
        """
        Reset the internal dictionary that keeps track of the parameters state.
        """
        restart_dict = self._initialize_params_dict()
        self.params_dict = restart_dict

    def _train(self, loader):
        self.model.train()

        losses = []  # loss as mean of batch losses

        for features, labels in loader:
            # forward pass
            out = self.model(features.to(self.device))

            # loss
            loss = self._compute_loss(out, labels.to(self.device))

            # remove gradient from previous passes
            self.optimizer.zero_grad()

            # backprop
            loss.backward()

            # parameters update
            self.optimizer.step()

            losses.append(loss.item())

        return statistics.mean(losses)

    def _validate(self, loader):
        self.model.eval()

        losses = []  # loss as mean of batch losses

        with torch.no_grad():
            for features, labels in loader:
                out = self.model(features.to(self.device))
                loss = self._compute_loss(out, labels.to(self.device))

                losses.append(loss.item())

        return statistics.mean(losses)

    def _compute_loss(self, real, target):
        try:
            loss = self.criterion(real, target)
        except:
            loss = self.criterion(real, target.long())
            msg = f"Target tensor has been casted from"
            msg = f"{msg} {type(target)} to 'long' dtype to avoid errors."
            warnings.warn(msg)

        # apply regularization if any
        if self.regularizer is not None:
            penalty = self.regularizer(self.model.named_parameters())
            loss += penalty.item()

        return loss

    def _get_device(self, device):
        if device is None:
            dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            msg = f"Device was automatically selected: {dev}"
            warnings.warn(msg)
        else:
            dev = device

        return dev
