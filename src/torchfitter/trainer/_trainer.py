import time
import torch
import logging
import warnings
import statistics
import torchmetrics

from tqdm.auto import tqdm
from typing import List
from torchfitter.callbacks.base import CallbackHandler
from torchfitter.conventions import ParamsDict, BarFormat
from torchfitter.trainer._utils import TrainerInternalState, MetricsHandler


class Trainer:
    """Class that eases the training of a PyTorch model.
    
    The trainer tracks its state using a 
    'torchfitter.trainer.TrainerInternalState' object. You can also pass a list
    of callbacks and/or metrics to the trainer. The callbacks will be runned
    at different points depending on the methods that were filled. The metrics
    will be runned in the train and validation steps.

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
    mixed_precision : bool, optional, default: False
        Whether to use mixed precision training or not. If True, the forward 
        pass will be computed under the context of `torch.cuda.amp.autocast` 
        and the backpropagation and gradient descent steps will be computed 
        using `torch.cuda.amp.GradScaler`.
        Callbacks that allow interaction.
    metrics : list of torchmetrics.Metric, optional, default: None
        List of metrics to compute in the fitting process.

    Attributes
    ----------
    callback_handler : torchfitter.callback.CallbackHandler
        Handles the passed callbacks.
    internal_state : torchfitter.trainer.TrainerInternalState
        Trainer internal parameters state.
    metrics_handler : torchfitter.trainer.MetricsHandler
        Handles the passed metrics.

    Warning
    -------
    The trainer class will automatically select the device if is None, which 
    will may cause problems when tensors and/or modules are on different 
    devices.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        criterion: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        regularizer=None,
        device=None,
        mixed_precision: bool=False,
        callbacks: list=None,
        metrics: List[torchmetrics.Metric]=None
    ):
        self.criterion = criterion
        self.optimizer = optimizer
        self.regularizer = regularizer
        self.device = self._get_device(device)
        self.model = model.to(self.device)
        self.callbacks_list = callbacks
        self.metrics_list = metrics
        self.mixed_precision = mixed_precision

        # attributes
        self.internal_state = TrainerInternalState(
            model=self.model, device=self.device
        )
        self.callback_handler = CallbackHandler(
            callbacks_list=self.callbacks_list
        )
        self.metrics_handler = MetricsHandler(metrics_list=self.metrics_list)
        self.__bar_format = BarFormat.FORMAT
        self.__scaler = torch.cuda.amp.GradScaler(enabled=self.mixed_precision)

        logging.basicConfig(level=logging.INFO)

        if self.metrics_handler.metric_names is not None:
            names = self.metrics_handler.metric_names
            self.internal_state.add_metrics(*names)

    def fit(
        self,
        train_loader: torch.utils.data.dataloader.DataLoader,
        val_loader: torch.utils.data.dataloader.DataLoader,
        epochs: int,
        disable_pbar: bool=False,
    ) -> None:
        """Fit the model.

        Fit the model using the given loaders for the given number of epochs. A
        progress bar will be displayed using tqdm unless 'disable_pbar' is set
        to True.

        Parameters
        ----------
        train_loader : torch.DataLoader
            DataLoader containing train dataset.
        val_loader : torch.DataLoader
            DataLoader containing validation dataset.
        epochs : int
            Number of training epochs.
        disable_pbar : bool, optional, default: False
            If True, the progress bar will be disabled.

        """
        initial_epoch = self.internal_state.get_single_param(
            key=ParamsDict.EPOCH_NUMBER
        )
        n_iter = len(train_loader) + len(val_loader)

        # track total training time
        total_start_time = time.time()

        self.callback_handler.on_fit_start(self.internal_state.get_state_dict())

        # ---- fitting process ----
        epoch = initial_epoch
        stop = self.internal_state.get_single_param(key=ParamsDict.STOP_TRAINING)
        while epoch <= epochs and not stop:
            with tqdm(
                range(1, n_iter+1),
                bar_format=self.__bar_format,
                ascii=True,
                leave=False,
                disable=disable_pbar,
                unit=' batch'
            ) as pbar:
                pbar.set_description(f"Epoch: {epoch}/{epochs}")
                
                self.internal_state.update_params(**{ParamsDict.PROG_BAR: pbar})
                self.callback_handler.on_epoch_start(self.internal_state.get_state_dict())

                # track epoch time
                epoch_start_time = time.time()

                # train
                self.callback_handler.on_train_step_start(self.internal_state.get_state_dict())
                tr_loss = self.train_step(train_loader)
                self.callback_handler.on_train_step_end(self.internal_state.get_state_dict())

                # validation
                self.callback_handler.on_validation_step_start(self.internal_state.get_state_dict())
                val_loss = self.validation_step(val_loader)
                self.callback_handler.on_validation_step_end(self.internal_state.get_state_dict())

                self.internal_state.update_history(
                    **{
                        ParamsDict.HISTORY_TRAIN_LOSS: tr_loss,
                        ParamsDict.HISTORY_VAL_LOSS: val_loss,
                        ParamsDict.HISTORY_LR: self.optimizer.param_groups[0]["lr"],
                    }
                )

                epoch_time = time.time() - epoch_start_time
                self.internal_state.update_params(
                    **{
                        ParamsDict.VAL_LOSS: val_loss,
                        ParamsDict.TRAIN_LOSS: tr_loss,
                        ParamsDict.EPOCH_TIME: epoch_time,
                        ParamsDict.EPOCH_NUMBER: epoch,
                        ParamsDict.TOTAL_EPOCHS: epochs,
                    }
                )

                self.callback_handler.on_epoch_end(self.internal_state.__dict__)
                
                epoch += 1
                stop = self.internal_state.get_single_param(key=ParamsDict.STOP_TRAINING)

                if not disable_pbar:
                    pbar.reset() # DISC: pbar.close() ??

        total_time = time.time() - total_start_time

        self.internal_state.update_params(**{ParamsDict.TOTAL_TIME: total_time})
        self.callback_handler.on_fit_end(self.internal_state.__dict__)

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

    def set_scaler(
        self, scaler: torch.cuda.amp.grad_scaler.GradScaler
    ) -> None:
        """Set the gradient scaler used in mixed precision.

        The trainer creates a gradient scaler by default with the default 
        values for the constructor arguments except 'enabled', which will be 
        set the same as 'mixed_precision' variable value.

        Parameters
        ----------
        scaler : torch.cuda.amp.grad_scaler.GradScaler
            The gradient scaler you want to use.
        """
        self.__scaler = scaler

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
        restart_dict = self.internal_state.reset_parameters(
            reset_model=reset_model
        )

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
        """Perform a train step using the given dataloader.

        A train step consists of running and optimizing the model for each 
        batch in the given train dataloader.

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
        for features, labels in loader:

            labels = labels.to(self.device)
            features = features.to(self.device)
            
            # TODO: find cleaner way to do this
            if self.mixed_precision:
                with torch.cuda.amp.autocast():
                    # forward pass and loss
                    out = self.model(features)
                    loss = self.compute_loss(out, labels)

                # clean gradients, backpropagation and parameters update
                self.optimizer.zero_grad()
                self.__scaler.scale(loss).backward()
                self.__scaler.step(self.optimizer)
                self.__scaler.update()
            else:
                out = self.model(features)
                loss = self.compute_loss(out, labels)

                # clean gradients, backpropagation and parameters update
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            # compute metrics, needed for accumulated computation
            _ = self.metrics_handler.single_batch_computation(
                predictions=out, target=labels
            )

            losses.append(loss.item())
            self.internal_state.update_progress_bar(n=1)

        # compute accumulated metrics (metric.compute())
        metrics_accumulated = self.metrics_handler.accumulated_batch_computation()
        
        if metrics_accumulated is not None:
            self.internal_state.update_metrics(is_train=True, **metrics_accumulated)

        # reset metrics
        self.metrics_handler.reset_metrics()

        return statistics.mean(losses)

    def validation_step(
        self, loader: torch.utils.data.dataloader.DataLoader
    ) -> float:
        """Perform a validation step using the given dataloader.

        A validation step consists of running and the model for each batch in 
        the given validation dataloader.

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
            for features, labels in loader:
                # will be used multiple times
                labels = labels.to(self.device)
                features = features.to(self.device)

                out = self.model(features)
                loss = self.compute_loss(out, labels)

                # compute metrics, needed for accumulated computation
                _ = self.metrics_handler.single_batch_computation(
                    predictions=out, target=labels
                )

                losses.append(loss.item())
                self.internal_state.update_progress_bar(n=1)

            # compute accumulated metrics
            metrics_accumulated = self.metrics_handler.accumulated_batch_computation()
            
            if metrics_accumulated is not None:
                self.internal_state.update_metrics(is_train=False, **metrics_accumulated)

            # reset metrics
            self.metrics_handler.reset_metrics()

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
            Loss graph contained in a (1 x 1) torch.Tensor.

        Warning
        -------
        This method will cast the target tensor to 'long' data type if needed.
        """
        try:
            loss = self.criterion(real, target)
        except:
            loss = self.criterion(real, target.long())
            msg = (
                f"""Target tensor has been casted from"""
                """{target.dtype} to 'long' dtype to avoid errors."""
            )
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
