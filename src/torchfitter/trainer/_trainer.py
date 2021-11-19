import time
import torch
import logging
import warnings
import statistics
import torchmetrics
from tqdm.auto import tqdm
from typing import List, Tuple
from accelerate import Accelerator
from torchfitter.callbacks.base import CallbackHandler
from torchfitter.conventions import ParamsDict, BarFormat
from torchfitter.trainer._utils import TrainerInternalState, MetricsHandler


class Trainer:
    """Class that eases the training of a PyTorch model.

    This class leverages the power of 'accelerate' to handle the device 
    management and the model optimization.
    
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
    mixed_precision : bool, optional, default: False
        Whether to use mixed precision training or not. If True, the forward 
        pass will be computed under the context of `torch.cuda.amp.autocast` 
        and the backpropagation and gradient descent steps will be computed 
        using `torch.cuda.amp.GradScaler`.
    callbacks : list of torchfitter.callback.Callback
        Callbacks to use during the training process.
    metrics : list of torchmetrics.Metric, optional, default: None
        List of metrics to compute in the fitting process.
    accelerator : accelerate.Accelerator
        Accelerator object from 'accelerate'.
    accumulate_iter : int, optional, default: 1
        Accumulate gradients every 'accumulate_iter' iterations.
    gradient_clipping : str or None, optional, {None, 'norm', 'value'}
        Norm gradient clipping of value gradient clipping. If None, gradient
        clipping won't be applied.
    gradient_clipping_kwrgs : dict, optional, default: None
        Dictionary containing keyword arguments for gradient clipping 
        algorithm. Example: {max_norm=1, norm_type=2}. See 
        https://huggingface.co/docs/accelerate/accelerator.html for more 
        information.
    log_level : int, optional, default: logging.INFO
        Logging level.

    Attributes
    ----------
    callback_handler : torchfitter.callback.CallbackHandler
        Handles the passed callbacks.
    internal_state : torchfitter.trainer.TrainerInternalState
        Trainer internal parameters state.
    metrics_handler : torchfitter.trainer.MetricsHandler
        Handles the passed metrics.
    gradient_clipping_algo_ : callable
        Gradient clipping algorithm or None

    """

    def __init__(
        self,
        model: torch.nn.Module,
        criterion: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        regularizer=None,
        mixed_precision: bool=False,
        callbacks: list=None,
        metrics: List[torchmetrics.Metric]=None,
        accelerator: Accelerator=None,
        accumulate_iter: int=1,
        gradient_clipping: str=None,
        gradient_clipping_kwrgs: dict=None,
        log_level: int=logging.INFO
    ):
        self.criterion = criterion
        self.regularizer = regularizer
        self.callbacks_list = callbacks
        self.metrics_list = metrics
        self.accumulate_iter = accumulate_iter
        self.gradient_clipping = gradient_clipping
        self.gradient_clipping_kwrgs = gradient_clipping_kwrgs
        self.log_level = log_level

        if accelerator is None:
            self.accelerator = Accelerator(fp16=mixed_precision)

        # wrap withing accelerate environment
        self.optimizer = self.accelerator.prepare_optimizer(optimizer)
        self.model = self.accelerator.prepare_model(model)

        # ----- attributes -----
        self.internal_state = TrainerInternalState(
            model=self.model, accelerator=self.accelerator
        )
        self.callback_handler = CallbackHandler(
            callbacks_list=self.callbacks_list
        )
        self.metrics_handler = MetricsHandler(metrics_list=self.metrics_list)
        self.gradient_clipping_algo_ = self._prepare_gradient_clipping()
        self.__bar_format = BarFormat.FORMAT

        if self.metrics_handler.metric_names is not None: # TODO: do this inside MetricsHandler
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
            If True, the TQDM progress bar will be disabled.

        """
        initial_epoch = self.internal_state.get_single_param(
            key=ParamsDict.EPOCH_NUMBER
        )
        n_iter = len(train_loader) + len(val_loader)

        # wrap loaders within accelerate environment
        train_loader = self.accelerator.prepare_data_loader(train_loader)
        val_loader = self.accelerator.prepare_data_loader(val_loader)

        # track total training time
        total_start_time = time.time()

        self.callback_handler.on_fit_start(self.internal_state.get_state_dict())

        # ---- fitting process ----
        epoch = initial_epoch
        stop = False
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
                self.callback_handler.on_epoch_start(
                    self.internal_state.get_state_dict()
                )

                # track epoch time
                epoch_start_time = time.time()

                # train step
                self.callback_handler.on_train_step_start(
                    self.internal_state.get_state_dict()
                )
                tr_loss = self.train_step(train_loader) # actual step
                self.callback_handler.on_train_step_end(
                    self.internal_state.get_state_dict()
                )

                # validation step
                self.callback_handler.on_validation_step_start(
                    self.internal_state.get_state_dict()
                )
                val_loss = self.validation_step(val_loader)
                self.callback_handler.on_validation_step_end(
                    self.internal_state.get_state_dict()
                )

                self.internal_state.update_history(
                    **{
                        ParamsDict.HISTORY_TRAIN_LOSS: tr_loss,
                        ParamsDict.HISTORY_VAL_LOSS: val_loss,
                        ParamsDict.HISTORY_LR: 
                            self.optimizer.param_groups[0]["lr"],
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

                self.callback_handler.on_epoch_end(
                    self.internal_state.get_state_dict()
                )
                
                epoch += 1
                stop = self.internal_state.get_single_param(
                    key=ParamsDict.STOP_TRAINING
                )

                if not disable_pbar:
                    pbar.reset() # DISC: pbar.close() ??

        total_time = time.time() - total_start_time

        self.internal_state.update_params(**{ParamsDict.TOTAL_TIME: total_time})
        self.callback_handler.on_fit_end(self.internal_state.get_state_dict())

    def _prepare_gradient_clipping(self):
        """
        Identify the gradient clipping algorithm to use.

        Returns
        -------
        algo : callable
            Callable function that wraps the gradient clipping funcionality.
        """
        if self.gradient_clipping == 'value':
            algo = self.accelerator.clip_grad_value_
        elif self.gradient_clipping == 'norm':
            algo = self.accelerator.clip_grad_norm_
        elif self.gradient_clipping is None:
            algo = None
        else:
            raise ValueError(
                "Not supported gradient "
                f"gradient clipping algorithm: '{self.gradient_clipping}'"
            )
        return algo

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

        Since the trainer relies on accelerate.Accelator class for the fitting
        process the scaler is created inside the aforementioned. With this 
        function you can set the scaler to be an instance with the desired 
        argument values.

        Parameters
        ----------
        scaler : torch.cuda.amp.grad_scaler.GradScaler
            The gradient scaler you want to use.
        """
        self.accelerator.scaler = scaler

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
            if self.callbacks_list is not None:
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
        for batch_idx, batch in enumerate(loader):
            loss = self.batch_train_step(batch_index=batch_idx, batch=batch)
            losses.append(loss.item())
            self.internal_state.update_progress_bar(n=1)

        # compute accumulated metrics (metric.compute())
        metrics_accumulated = self.metrics_handler.accumulated_batch_computation()
        
        if metrics_accumulated is not None:
            self.internal_state.update_metrics(
                is_train=True, **metrics_accumulated
            )

        # reset metrics
        self.metrics_handler.reset_metrics()
        return statistics.mean(losses)

    def batch_train_step(
        self, batch_index, batch: Tuple[torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        """
        Define the computations to perform on each batch for the training loop.

        Parameters
        ----------
        batch_index : int
            Batch index.
        batch : tuple
            Batch to process.

        Returns
        -------
        loss : torch.Tensor
            Train loss graph.
        """
        features, labels = batch

        # forward propagation
        out = self.model(features)
        loss = self.compute_loss(out, labels) / self.accumulate_iter

        # backpropagation
        self.accelerator.backward(loss)

        # gradient clipping
        if self.gradient_clipping_algo_ is not None:
            self.gradient_clipping_algo_(
                self.model.parameters(), **self.gradient_clipping_kwrgs
            )
        
        # gradient accumulation logic
        if (batch_index + 1) % self.accumulate_iter == 0: # or batch_index + 1 == len(loader)
            # update parameters and remove gradient
            self.optimizer.step()
            self.optimizer.zero_grad()

        # compute metrics, needed for accumulated computation
        _ = self.metrics_handler.single_batch_computation(
            predictions=out, target=labels
        )
        return loss

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
            for batch_idx, batch in enumerate(loader):
                loss = self.batch_validation_step(
                    batch_index=batch_idx, batch=batch
                )
                losses.append(loss.item())
                self.internal_state.update_progress_bar(n=1)

            # compute accumulated metrics
            metrics_accumulated = self.metrics_handler.accumulated_batch_computation()
            
            if metrics_accumulated is not None:
                self.internal_state.update_metrics(
                    is_train=False, **metrics_accumulated
                )

            # reset metrics
            self.metrics_handler.reset_metrics()

        return statistics.mean(losses)

    def batch_validation_step(
        self, batch_index, batch: Tuple[torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        """
        Define the computations to perform on each batch for the validation 
        loop.

        Parameters
        ----------
        batch_index : int
            Batch index.
        batch : tuple
            Batch to process.

        Returns
        -------
        loss : torch.Tensor
            Validation loss graph.
        """
        features, labels = batch

        out = self.model(features)
        loss = self.compute_loss(out, labels)

        # compute metrics, needed for accumulated computation
        _ = self.metrics_handler.single_batch_computation(
            predictions=out, target=labels
        )
        return loss

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
                f"""{target.dtype} to 'long' dtype to avoid errors."""
            )
            warnings.warn(msg)

        # apply regularization if any
        if self.regularizer is not None:
            penalty = self.regularizer(
                self.model.named_parameters(), self.accelerator.device
            )
            loss += penalty.item()

        return loss

    def save_model(self, path):
        """
        Convenient method to save the model ensuring the model is unwrapped and
        all processes are done.

        Parameters
        ----------
        path : path-like
            Path to save the model.
        """
        self.accelerator.wait_for_everyone()
        unwrapped_model = self.accelerator.unwrap_model(self.model)
        self.accelerator.save(unwrapped_model.state_dict(), path)

    def load_model(self, path):
        """
        Convenient method to load the model ensuring the model is unwrapped.

        Parameters
        ----------
        path : path-like
            Path to load the model from.
        """
        unwrapped_model = self.accelerator.unwrap_model(self.model)
        unwrapped_model.load_state_dict(torch.load(path))
        self.model = unwrapped_model
