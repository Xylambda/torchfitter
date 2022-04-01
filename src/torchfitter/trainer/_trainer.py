import logging
import statistics
import time
from typing import List, Tuple, Union

import torch
import torchmetrics
from accelerate import Accelerator
from numpy import ndarray
from torch.utils.data.dataloader import DataLoader

from torchfitter.callbacks.base import Callback, CallbackHandler
from torchfitter.conventions import ParamsDict
from torchfitter.regularization.base import RegularizerBase
from torchfitter.trainer._utils import MetricsHandler, TrainerInternalState


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
        Loss function criterion used to optimize the model.
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
        List of metrics to compute in the fitting process. The metrics will be
        registered in the internal state using the name of the class. For
        example, passing `[MeanSquaredError()]` will be registered as
        `MeanSquaredError`.
    accelerator : accelerate.Accelerator
        Accelerator object from 'accelerate'. If no object is passed, the
        trainer will create an instance with the default parameters.
    accumulate_iter : int, optional, default: 1
        Accumulate gradients every 'accumulate_iter' iterations. The default
        value does not accumulate the gradients.
    gradient_clipping : {None, 'norm', 'value'}
        Norm gradient clipping of value gradient clipping. If None, gradient
        clipping won't be applied.
    gradient_clipping_kwargs : dict, optional, default: None
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
        Gradient clipping algorithm or None.

    """

    def __init__(
        self,
        model: torch.nn.Module,
        criterion: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        regularizer: RegularizerBase = None,
        mixed_precision: bool = False,
        callbacks: List[Callback] = None,
        metrics: List[torchmetrics.Metric] = None,
        accelerator: Accelerator = None,
        accumulate_iter: int = 1,
        gradient_clipping: str = None,
        gradient_clipping_kwargs: dict = None,
        log_level: int = logging.INFO,
    ):
        self.criterion = criterion
        self.regularizer = regularizer
        self.callbacks_list = callbacks
        self.metrics_list = metrics
        self.accumulate_iter = accumulate_iter
        self.gradient_clipping = gradient_clipping
        self.gradient_clipping_kwargs = gradient_clipping_kwargs
        self.log_level = log_level

        if accelerator is None:
            self.accelerator = Accelerator(fp16=mixed_precision)

        # wrap withing accelerate environment
        self.optimizer = self.accelerator.prepare_optimizer(optimizer)
        self.model = self.accelerator.prepare_model(model)

        # ----- attributes -----
        self.internal_state = TrainerInternalState(
            model=self.model, accelerator=self.accelerator, optimizer=optimizer
        )
        self.callback_handler = CallbackHandler(
            callbacks_list=self.callbacks_list
        )
        self.metrics_handler = MetricsHandler(
            metrics_list=self.metrics_list, criterion=criterion
        )
        self.gradient_clipping_algo_ = self._prepare_gradient_clipping()

        if self.metrics_handler.metric_names is not None:
            names = self.metrics_handler.metric_names
            self.internal_state.add_metrics(*names)

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int,
    ) -> dict:
        """Fit the model.

        Fit the model using the given loaders for the given number of epochs.
        By default, the trainer does not display any information about the
        fitting process, but you can use any of the callbacks designed for that
        purpose or create your own callbacks.

        Parameters
        ----------
        train_loader : torch.DataLoader
            DataLoader to iterate over the train dataset.
        val_loader : torch.DataLoader
            DataLoader to iterate over the validation dataset.
        epochs : int
            Number of training epochs.

        Returns
        -------
        history : dict
            Dictionay with epoch and batch metrics. The metrics contained in
            the dictionary will be the passed metrics + criterion results.

        """
        initial_epoch = self.internal_state.get_single_param(
            key=ParamsDict.EPOCH_NUMBER
        )

        # wrap loaders within accelerate environment
        train_loader = self.accelerator.prepare_data_loader(train_loader)
        val_loader = self.accelerator.prepare_data_loader(val_loader)

        # update internal state with loaders
        self.internal_state.update_params(
            **{
                ParamsDict.TRAIN_LOADER: train_loader,
                ParamsDict.VAL_LOADER: val_loader,
                ParamsDict.TOTAL_EPOCHS: epochs,
            }
        )

        # track total training time
        total_start_time = time.perf_counter()
        self.callback_handler.on_fit_start(self.state_dict())

        # ---- fitting process ----
        epoch = initial_epoch
        stop = False
        while epoch <= epochs and not stop:
            self.callback_handler.on_epoch_start(self.state_dict())

            # track epoch time
            epoch_start_time = time.perf_counter()

            # ------- train step -------
            self.callback_handler.on_train_step_start(self.state_dict())
            tr_loss = self.train_step(train_loader)  # actual step
            self.callback_handler.on_train_step_end(self.state_dict())

            # ------- validation step -------
            self.callback_handler.on_validation_step_start(self.state_dict())
            val_loss = self.validation_step(val_loader)
            self.callback_handler.on_validation_step_end(self.state_dict())

            # -------- update internal state to track training --------
            self.internal_state.update_lr_history(
                value=self.optimizer.param_groups[0]["lr"], is_batch=False
            )

            # synchronize before measuring time
            self.accelerator.wait_for_everyone()
            epoch_time = time.perf_counter() - epoch_start_time
            self.internal_state.update_params(
                **{
                    ParamsDict.VAL_LOSS: val_loss,
                    ParamsDict.TRAIN_LOSS: tr_loss,
                    ParamsDict.EPOCH_TIME: epoch_time,
                }
            )

            self.callback_handler.on_epoch_end(self.state_dict())

            epoch += 1
            stop = self.internal_state.get_single_param(
                key=ParamsDict.STOP_TRAINING
            )
            self.internal_state.update_params(
                **{ParamsDict.EPOCH_NUMBER: epoch}
            )

        total_time = time.perf_counter() - total_start_time

        self.internal_state.update_params(
            **{ParamsDict.TOTAL_TIME: total_time}
        )
        self.callback_handler.on_fit_end(self.state_dict())

        # construct history object to return
        history = {
            ParamsDict.EPOCH_HISTORY: self.internal_state.get_single_param(
                key=ParamsDict.EPOCH_HISTORY
            ),
            ParamsDict.BATCH_HISTORY: self.internal_state.get_single_param(
                key=ParamsDict.BATCH_HISTORY
            ),
        }
        return history

    @torch.no_grad()
    def predict(
        self,
        X: Union[torch.Tensor, ndarray],
        as_array=False,
        dtype: str = "float",
    ) -> Union[torch.Tensor, ndarray]:
        """
        Predict function.

        Parameters
        ----------
        X : torch.Tensor or numpy.ndarray
            Data to use to make inference.
        as_array : bool, optional, default: False
            Whether to output the predictions as a numpy.narray or not.
        dtype : str, optional, default: "float"
            Data type to cast input tensor to.

        Returns
        -------
        predictions : torch.Tensor or numpy.ndarray
            Predicted values.

        See Also
        --------
        torchfitter.trainer.predict_tensor
        """
        if isinstance(X, ndarray):
            _tensor = torch.from_numpy(X)
            tensor = getattr(_tensor, dtype)()

        self.callback_handler.on_predict_begin(self.state_dict())
        predictions = self.predict_tensor(tensor)
        self.callback_handler.on_predict_end(self.state_dict())

        if as_array:
            return predictions.cpu().numpy()
        else:
            return predictions

    def predict_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """Make prediction for a given torch tensor.

        The passed tensor will be moved to the device the accelerator chose at
        the beginning of the training process.

        Parameters
        ----------
        tensor : torch.Tensor
            Tensor to use to make inference.

        Returns
        -------
        torch.Tensor
            Predicted values.

        See Also
        --------
        torchfitter.trainer.predict
        """
        device = self.accelerator.device
        tensor = tensor.to(device)
        return self.model(tensor)

    def _prepare_gradient_clipping(self) -> callable:
        """
        Identify the gradient clipping algorithm to use.

        Returns
        -------
        algo : callable
            Callable function that wraps the gradient clipping funcionality.
        """
        if self.gradient_clipping == "value":
            algo = self.accelerator.clip_grad_value_

        elif self.gradient_clipping == "norm":
            algo = self.accelerator.clip_grad_norm_

        elif self.gradient_clipping is None:
            algo = None

        else:
            raise ValueError(
                "Not supported gradient "
                f"gradient clipping algorithm: '{self.gradient_clipping}'"
            )
        return algo

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

    def reset_parameters(self, reset_model: bool = False) -> None:
        """
        Reset the internal dictionary that keeps track of the parameters state.

        Parameters
        ----------
        reset_model : bool, optional, default: False
            True to reset the model state.
        """
        restart_dict = self.internal_state.reset_parameters(
            reset_model=reset_model
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
        epoch_loss : float
            Mean loss of the batch.
        """
        self.model.train()

        losses = []  # loss as mean of batch losses
        for batch_idx, batch in enumerate(loader):
            self.callback_handler.on_train_batch_start(self.state_dict())
            loss = self.batch_train_step(batch_index=batch_idx, batch=batch)
            self.callback_handler.on_train_batch_end(self.state_dict())

            losses.append(loss.item())

        # compute accumulated metrics (metric.compute())
        metrics_accumulated = (
            self.metrics_handler.accumulated_batch_computation()
        )

        if metrics_accumulated is not None:
            self.internal_state.update_metrics(
                is_train=True, is_batch=False, **metrics_accumulated
            )

        epoch_loss = statistics.mean(losses)
        self.internal_state.update_loss_history(
            value=epoch_loss, is_train=True, is_batch=False
        )

        # reset metrics
        self.metrics_handler.reset_metrics()
        return epoch_loss

    def batch_train_step(
        self,
        batch_index,
        batch: Tuple[torch.Tensor, torch.Tensor],
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
        # assume last tensor in batch are labels
        batch_len = len(batch)
        features, labels = batch[:batch_len-1], batch[-1]

        # forward propagation
        out = self.model(*features)
        loss = self.loss_step(out, labels) / self.accumulate_iter

        # backpropagation
        self.accelerator.backward(loss)

        # gradient clipping
        if self.gradient_clipping_algo_ is not None:
            self.gradient_clipping_algo_(
                self.model.parameters(), **self.gradient_clipping_kwargs
            )

        # gradient accumulation logic
        batch_idx_plus = batch_index + 1
        loader_len = self.internal_state.get_single_param(
            key=ParamsDict.TRAIN_LOADER
        )
        if (
            batch_idx_plus % self.accumulate_iter == 0
            or batch_idx_plus == loader_len
        ):
            # update parameters and remove gradient
            self.optimizer.step()
            self.optimizer.zero_grad()

        # compute metrics, needed for accumulated computation
        metrics_single = self.metrics_handler.single_batch_computation(
            predictions=out, target=labels
        )

        # ----------- internal state updated ------------
        if metrics_single is not None:
            self.internal_state.update_metrics(
                is_train=True, is_batch=True, **metrics_single
            )

        # update history
        self.internal_state.update_loss_history(
            value=loss.item(), is_train=True, is_batch=True
        )

        self.internal_state.update_lr_history(
            value=self.optimizer.param_groups[0]["lr"], is_batch=True
        )

        # update internal state
        self.internal_state.update_params(
            **{
                ParamsDict.TRAIN_BATCH: batch,
                ParamsDict.TRAIN_BATCH_IDX: batch_index,
            }
        )

        return loss

    @torch.no_grad()
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
        epoch_loss : float
            Mean loss of the batch.
        """
        self.model.eval()

        losses = []  # loss as mean of batch losses
        for batch_idx, batch in enumerate(loader):
            self.callback_handler.on_validation_batch_start(self.state_dict())
            loss = self.batch_validation_step(
                batch_index=batch_idx, batch=batch
            )
            self.callback_handler.on_validation_batch_end(self.state_dict())
            losses.append(loss.item())

        # compute accumulated metrics
        metrics_accumulated = (
            self.metrics_handler.accumulated_batch_computation()
        )

        if metrics_accumulated is not None:
            self.internal_state.update_metrics(
                is_train=False, **metrics_accumulated
            )

        epoch_loss = statistics.mean(losses)
        self.internal_state.update_loss_history(
            value=epoch_loss, is_train=False, is_batch=False
        )

        # reset metrics
        self.metrics_handler.reset_metrics()

        return epoch_loss

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
        # assume last tensor in batch are labels
        batch_len = len(batch)
        features, labels = batch[:batch_len-1], batch[-1]

        out = self.model(*features)
        loss = self.loss_step(out, labels)

        # compute metrics, needed for accumulated computation
        metrics_single = self.metrics_handler.single_batch_computation(
            predictions=out, target=labels
        )

        # ----------- internal state updated ------------
        # update metrics dict
        if metrics_single is not None:
            self.internal_state.update_metrics(
                is_train=False, is_batch=True, **metrics_single
            )

        # update internal state
        self.internal_state.update_loss_history(
            value=loss.item(), is_train=False, is_batch=True
        )

        self.internal_state.update_lr_history(
            value=self.optimizer.param_groups[0]["lr"], is_batch=True
        )

        self.internal_state.update_params(
            **{
                ParamsDict.VAL_BATCH: batch,
                ParamsDict.VAL_BATCH_IDX: batch_index,
            }
        )

        return loss

    def loss_step(
        self, real: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """Compute loss graph.

        If you override this method the passed regularizer algorithms won't be
        applied unless they are specifically added.

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
        """
        self.callback_handler.on_loss_step_begin(self.state_dict())
        loss = self.criterion(real, target)

        # TODO: update loss in internal state
        self.callback_handler.on_loss_step_end(self.state_dict())

        # apply regularization if any
        if self.regularizer is not None:
            penalty = self.regularizer(
                self.model.named_parameters(), self.accelerator.device
            )
            loss += penalty.item()

        return loss

    def save_model(self, path):
        """
        Convenient method to save the model ensuring it is unwrapped and
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
        Convenient method to load the model ensuring it is unwrapped.

        Parameters
        ----------
        path : path-like
            Path to load the model from.
        """
        unwrapped_model = self.accelerator.unwrap_model(self.model)
        unwrapped_model.load_state_dict(torch.load(path))
        self.model = unwrapped_model

    def state_dict(self) -> dict:
        """Return current state dict.

        The state dict will change as the training progresses.

        Returns
        -------
        state : dict
            A dictionary containing the current state of the trainer.
        """
        state = self.internal_state.get_state_dict()
        return state
