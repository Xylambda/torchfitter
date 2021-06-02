import time
import torch
import logging
import warnings
import statistics

from tqdm.auto import tqdm
import torchfitter
from torchfitter.conventions import ParamsDict
from torchfitter.callbacks.base import CallbackHandler


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
        model: torch.nn.Module,
        criterion: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        regularizer=None,
        device=None,
        callbacks: list=None,
    ):
        self.criterion = criterion
        self.optimizer = optimizer
        self.regularizer = regularizer
        self.device = self._get_device(device)
        self.model = model.to(self.device)
        self.callbacks_list = callbacks

        # attributes
        self.params_dict = self._initialize_params_dict()
        self.callback_handler = CallbackHandler(
            callbacks_list=self.callbacks_list
        )

        # ----- create bar format ------
        r_bar = '| {n_fmt}/{total_fmt} | {rate_noinv_fmt}{postfix}, ramaining_time: {remaining} s'
        left = "{l_bar}{bar}"
        bar_fmt = f"{left}{r_bar}"
        self.__bar_format = bar_fmt

        logging.basicConfig(level=logging.INFO)

    def fit(
        self,
        train_loader: torch.utils.data.dataloader.DataLoader,
        val_loader: torch.utils.data.dataloader.DataLoader,
        epochs: int
    ) -> None:
        """Fit the model.

        Fit the model using the given loaders for the given number of epochs. A
        progress bar will be displayed using tqdm.

        Parameters
        ----------
        train_loader : torch.DataLoader
            DataLoader containing train dataset.
        val_loader : torch.DataLoader
            DataLoader containing validation dataset.
        epochs : int
            Number of training epochs.

        """
        # define progress bar
        initial_epoch = self.params_dict[ParamsDict.EPOCH_NUMBER]
        pbar = tqdm(
            range(initial_epoch, epochs+1),
            ascii=True,
            unit=' epoch',
            bar_format=self.__bar_format,
            leave=False,
            disable=False,
        )
        self._update_params_dict(**{ParamsDict.PROG_BAR: pbar})

        # track total training time
        total_start_time = time.time()

        self.callback_handler.on_fit_start(self.params_dict)

        # ---- train process ----
        for epoch in pbar:

            self.callback_handler.on_epoch_start(self.params_dict)

            # track epoch time
            epoch_start_time = time.time()

            # train
            self.callback_handler.on_train_batch_start(self.params_dict)
            tr_loss = self.train_step(train_loader)
            self.callback_handler.on_train_batch_end(self.params_dict)

            # validation
            self.callback_handler.on_validation_batch_start(self.params_dict)
            val_loss = self.validation_step(val_loader)
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
                    ParamsDict.VAL_LOSS: val_loss,
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

    def _update_history(self, **kwargs):
        """
        Update history paramaters dictionary with the passed key-value pairs.

        Parameters
        ----------
        kwargs : dict
            Dictionary with keys to update.
        """
        for key, value in kwargs.items():
            self.params_dict[ParamsDict.HISTORY][key].append(value)

    def _initialize_params_dict(self):
        params_dict = {
            ParamsDict.TRAIN_LOSS: float('inf'),
            ParamsDict.VAL_LOSS: float('inf'),
            ParamsDict.EPOCH_TIME: 0,
            ParamsDict.EPOCH_NUMBER: 1,
            ParamsDict.TOTAL_EPOCHS: None,
            ParamsDict.TOTAL_TIME: 0,
            ParamsDict.STOP_TRAINING: False,
            ParamsDict.DEVICE: self.device,
            ParamsDict.MODEL: self.model,
            ParamsDict.PROG_BAR: None,
            ParamsDict.HISTORY: {
                ParamsDict.HISTORY_TRAIN_LOSS: [],
                ParamsDict.HISTORY_VAL_LOSS: [],
                ParamsDict.HISTORY_LR: [],
            },
        }

        return params_dict

    def set_bar_format(self, fmt: str) -> None:
        """
        Set the bar format for the tqdm progress bar. See References for more
        info.

        Parameters
        ----------
        fmt : str
            New bar format.

        References
        ----------
        .. [1] tqdm API
           https://tqdm.github.io/docs/tqdm/
        """
        self.__bar_format = fmt

    def reset_parameters(
        self, reset_callbacks=False, reset_model=False
    ) -> None:
        """
        Reset the internal dictionary that keeps track of the parameters state.

        Parameters
        ----------
        reset_callbacks : bool, optional, default: False
            True to reset the callbacks states as well as the Callback Handler.
        reset_model : bool, optional, default: False
            True to reset the model state.
        """
        restart_dict = self._initialize_params_dict()
        
        if reset_model:
            restart_dict[ParamsDict.MODEL].reset_parameters()

        if reset_callbacks:
            if self.callbacks_list is None:
                pass
            else:
                for callback in self.callbacks_list:
                    callback.reset_parameters()

                self.callback_handler = CallbackHandler(
                    callbacks_list=self.callbacks_list
                )

        self.params_dict = restart_dict

    def train_step(
        self, loader: torch.utils.data.dataloader.DataLoader
    ) -> float:
        """Train step.

        Perform a train step using the given dataloader.

        Parameters
        ----------
        loader: torch.utils.data.dataloader.DataLoader
            Dataloader for train set.

        Returns
        -------
        float
            Mean loss of the batch.
        """
        self.model.train()

        losses = []  # loss as mean of batch losses

        for idx, (features, labels) in enumerate(loader):
            # forward pass and loss
            out = self.model(features.to(self.device))
            loss = self.compute_loss(out, labels.to(self.device))
            
            # clean gradients, backpropagation and params. update
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            losses.append(loss.item())

        return statistics.mean(losses)

    def validation_step(
        self, loader: torch.utils.data.dataloader.DataLoader
    ) -> float:
        """Validation step.

        Perform a validation step using the given dataloader.

        Parameters
        ----------
        loader: torch.utils.data.dataloader.DataLoader
            Dataloader for validation set.

        Returns
        -------
        float
            Mean loss of the batch.
        """
        self.model.eval()

        losses = []  # loss as mean of batch losses

        with torch.no_grad():
            for idx, (features, labels) in enumerate(loader):
                out = self.model(features.to(self.device))
                loss = self.compute_loss(out, labels.to(self.device))

                losses.append(loss.item())

        return statistics.mean(losses)

    def compute_loss(
        self, real: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """Compute loss graph.

        When this method is overrided, the regularization procedures that are 
        not included in the optimizer must be handled by the user.

        Parameters
        ----------
        real : torch.Tensor
            Obtained tensor after performing a forward pass.
        target : torch.Tensor
            Target tensor to compute loss.

        Returns
        -------
        loss : torch.Tensor
            Loss graph as (1x1) torch.Tensor.
        """
        try:
            loss = self.criterion(real, target)
        except:
            loss = self.criterion(real, target.long())
            msg = f"Target tensor has been casted from"
            msg = f"{msg} {type(target)} to 'long' dtype to avoid errors."
            warnings.warn(msg)

        # apply regularization if any
        if self.regularizer is not None:
            penalty = self.regularizer(
                self.model.named_parameters(), self.device
            )
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
