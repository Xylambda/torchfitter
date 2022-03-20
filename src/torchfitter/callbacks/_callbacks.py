""" Callbacks for the manager class """
import torch
import subprocess
from typing import List
from .base import Callback
from torchfitter.conventions import ParamsDict
from torch.optim.swa_utils import AveragedModel
from rich.progress import (
    Progress,
    BarColumn,
    TimeRemainingColumn,
)


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
    path : str or Path, optional, default: 'checkpoint.pt'
        Path to store the best observed parameters.
    """

    def __init__(self, patience=50, load_best=True, path="checkpoint.pt"):
        super(EarlyStopping, self).__init__()
        self.path = path
        self.patience = patience
        self.load_best = load_best

        self.log_name = "EarlyStopping"

    def __repr__(self) -> str:
        return f"EarlyStopping(patience={self.patience}, load_best={self.load_best})"

    def on_fit_start(self, params_dict):
        self.wait = 0
        self.stopped_epoch = 0
        self.best = float("inf")

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
                    self.logger.info("Best observed parameters loaded.")

    def on_fit_end(self, params_dict):
        if self.stopped_epoch > 0:
            self.logger.info(
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
    precision : int, optional, default: 2
        Number of decimals in the numbers.
    """

    def __init__(self, update_step, precision=2):
        super(LoggerCallback, self).__init__()
        self.update_step = update_step
        self.prec = precision

        self.log_name = "LoggerCallback"

    def on_fit_start(self, params_dict):
        dev = params_dict[ParamsDict.DEVICE]
        self.logger.info(f"Starting training process on {dev}")

    def on_epoch_end(self, params_dict) -> None:
        epoch_number = params_dict[ParamsDict.EPOCH_NUMBER]
        total_epochs = params_dict[ParamsDict.TOTAL_EPOCHS]
        val_loss = params_dict[ParamsDict.VAL_LOSS]
        train_loss = params_dict[ParamsDict.TRAIN_LOSS]
        epoch_time = params_dict[ParamsDict.EPOCH_TIME]

        prec = self.prec
        msg = (
            f"Epoch {epoch_number}/{total_epochs} | Train loss: "
            f"{train_loss:.{prec}e} | Validation loss {val_loss:.{prec}e} | "
            f"Time/epoch: {epoch_time:.{prec}e} s"
        )

        if epoch_number % self.update_step == 0 or epoch_number == 1:
            self.logger.info(msg)

    def on_fit_end(self, params_dict):
        total_time = params_dict[ParamsDict.TOTAL_TIME]
        # final message
        self.logger.info(
            f"""End of training. Total time: {total_time:0.5f} seconds"""
        )


class LearningRateScheduler(Callback):
    """Callback to schedule learning rate.

    `LearningRateScheduler` provides an easy abstraction to schedule the
    learning rate of the optimizer by calling `scheduler.step()` after the
    training step has been performed; i.e., `on_train_step_end`.

    If you were to use a learning rate scheduler in addition to stochastic
    averaging, you must pass both to an instance of `StochasticWeightAveraging`
    callback instance of creating an instance of `LearningRateScheduler`.

    Parameters
    ----------
    scheduler : torch.optim.lr_scheduler._LRScheduler
        Scheduler use to perform the lr reduction.
    metric : str, optional, default : torchfitter.conventions.ParamsDict.LOSS
        Metric to track in order to reduce the learning rate. If you want to
        use one of the passed metrics you must pass the class name. See
        examples section.
    on_train : bool, optional, default: True
        Whether to watch the train version of the metric (True) or the
        validation version of the metric (False)

    Examples
    --------
    Assume we already have the `kwargs` for the trainer and the necessary
    imports:

    >>> import torchmetrics
    >>> from torch.optim.lr_scheduler import ReduceLROnPlateau
    >>> from torchfitter.callbacks import LearningRateScheduler
    >>> sch = ReduceLROnPlateau(optimizer, factor=0.1, patience=50)
    >>> lr_sch = LearningRateScheduler(scheduler=sch, metric='MeanSquaredError', on_train=False)
    >>> metrics = [torchmetrics.MeanSquaredError]
    >>> trainer = Trainer(callbacks=[lr_sch], metrics=metrics, **kwargs)

    References
    ----------
    ..[1] PyTorch - How to adjust learning rate
       https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate

    """

    def __init__(
        self,
        scheduler: torch.optim.lr_scheduler._LRScheduler,
        metric: str = ParamsDict.LOSS,
        on_train: bool = True,
    ):
        super(LearningRateScheduler, self).__init__()

        self.scheduler = scheduler
        self.metric = metric
        self.on_train = on_train

    def __repr__(self) -> str:
        sch = type(self.scheduler).__name__
        return f"LearningRateScheduler(scheduler={sch}, metric={self.metric})"

    def on_train_step_end(self, params_dict: dict) -> None:
        if self.metric is not None:
            key = "train" if self.on_train else "validation"
            epoch_hist = params_dict[ParamsDict.EPOCH_HISTORY]
            metric = epoch_hist[self.metric][key][-1]
            self.scheduler.step(metric)
        else:
            self.scheduler.step()


class GPUStats(Callback):
    """GPU stats logger.

    The list of available queries can be found on NVIDIA smi queries. See
    `Notes` section for more information.

    Parameters
    ----------
    queries : list of str
        List of queries to log
    format : str
        Queries format.
    update_step : int, optional, default: 50
        Logs will be performed every 'update_step'.

    Notes
    -----
    To check the list of available queries, see
    https://nvidia.custhelp.com/app/answers/detail/a_id/3751/~/useful-nvidia-smi-queries

    """

    def __init__(
        self,
        format: str = "csv, nounits, noheader",
        queries: List[str] = [
            "name",
            "temperature.gpu",
            "utilization.gpu",
            "utilization.memory",
            "memory.used",
        ],
        update_step=50,
    ):
        super(GPUStats, self).__init__()

        self.queries = queries
        self.format = format
        self.update_step = update_step

        self.log_name = "GPU Stats"

    def on_epoch_end(self, params_dict):
        epoch_number = params_dict[ParamsDict.EPOCH_NUMBER]

        if epoch_number == 1 or epoch_number % self.update_step == 0:
            stdout = self._get_queries(
                queries=self.queries, format=self.format
            )
            msg = " | ".join(map(str, stdout))  # unpack and format
            self.logger.info(msg)

    def _get_queries(self, queries, format):
        stdout = []

        for query in queries:
            out = (
                subprocess.run(
                    f"nvidia-smi --query-gpu={query} --format={format}",
                    stdout=subprocess.PIPE,
                    encoding="utf-8",
                )
                .stdout.replace("\r", " ")
                .replace("\n", " ")
            )

            stdout.append(out)

        return stdout


class RichProgressBar(Callback):
    """
    This callback displays a progress bar to report the state of the training
    process: on each epoch, a new bar will be created and stacked below the
    previous bars.

    Metrics are logged using the library logger.

    Parameters
    ----------
    display_step : int
        Number of epochs to wait to display the progress bar.
    precision : int, optional, default: 2
        Number of decimals to use in the log.
    log_lr : bool, optional, default: False
        Whether to log the learning rate (True) or not (False).
    """

    def __init__(
        self, display_step: int = 1, log_lr: bool = False, precision: int = 2
    ):
        super(RichProgressBar, self).__init__()

        self.display_step = display_step
        self.prec = precision
        self.log_lr = log_lr

        self.log_name = "Rich Bar"

    def on_train_batch_end(self, params_dict: dict) -> None:
        epoch = params_dict[ParamsDict.EPOCH_NUMBER]
        if epoch % self.display_step == 0 or epoch == 1:
            self.progress_bar.advance(self.epoch_task, 1)

    def on_validation_batch_end(self, params_dict: dict) -> None:
        epoch = params_dict[ParamsDict.EPOCH_NUMBER]
        if epoch % self.display_step == 0 or epoch == 1:
            # advance bar
            self.progress_bar.advance(self.epoch_task, 1)

    def on_epoch_start(self, params_dict: dict) -> None:
        # gather necessary objects
        train_loader = params_dict[ParamsDict.TRAIN_LOADER]
        val_loader = params_dict[ParamsDict.VAL_LOADER]
        epoch = params_dict[ParamsDict.EPOCH_NUMBER]
        total_epochs = params_dict[ParamsDict.TOTAL_EPOCHS]

        # compute number of batches
        n_elements = len(train_loader) + len(val_loader)

        if epoch % self.display_step == 0 or epoch == 1:
            self.progress_bar = Progress(
                "[progress.description]{task.description}",
                "•",
                BarColumn(),
                "•",
                "[progress.percentage]{task.percentage:>3.0f}%",
                "•",
                TimeRemainingColumn(),
            )

            self.epoch_task = self.progress_bar.add_task(
                description=f"Epoch {epoch}/{total_epochs}",
                total=n_elements,
            )
            self.progress_bar.start()

    def on_epoch_end(self, params_dict: dict) -> None:
        epoch = params_dict[ParamsDict.EPOCH_NUMBER]

        if epoch % self.display_step == 0 or epoch == 1:
            # update metrics
            text = self.render_text(params_dict[ParamsDict.EPOCH_HISTORY])
            self.logger.info(text)
            self.progress_bar.stop()

    def render_text(self, update_dict):
        text_format = ""

        for metric in update_dict:
            if metric != ParamsDict.HISTORY_LR:
                train_metric = update_dict[metric]["train"][-1]
                val_metric = update_dict[metric]["validation"][-1]

                if text_format:  # not empty
                    text_format = (
                        f"{text_format} • {metric} > Train: "
                        f"{train_metric:.{self.prec}e} | "
                        f"Validation: {val_metric:.{self.prec}e}"
                    )
                else:
                    text_format = (
                        f"{metric} > Train: "
                        f"{train_metric:.{self.prec}e} | Validation: "
                        f"{val_metric:.{self.prec}e}"
                    )
            else:
                if self.log_lr:
                    text_format = (
                        f"{text_format} • LearningRate: "
                        f"{update_dict[metric][-1]}"
                    )

        return text_format


class StochasticWeightAveraging(Callback):
    """Applies a stochastic weight averaging to the training process.

    If you were to use a learning rate scheduler in addition to stochastic
    averaging, you must pass both to the constructor of this class instead of
    creating an individual callback for the standard lr scheduler. See
    `Examples` section.

    Parameters
    -----------
    swa_scheduler : torch.optim.lr_scheduler.Scheduler
        Stochastic weight averaging scheduler.
    start_epoch : int
        Epoch from which to start applying stochastic weight average.
    scheduler : torch.optim.lr_scheduler._LRScheduler
        Scheduler use to perform the lr reduction.
    metric : str, optional, default : torchfitter.conventions.ParamsDict.LOSS
        Metric to track in order to reduce the learning rate. If you want to
        use one of the passed metrics you must pass the class name. See
        examples section.
    on_train : bool, optional, default: True
        Whether to watch the train version of the metric (True) or the
        validation version of the metric (False)

    References
    ----------
    .. [1] Pavel Izmailov, Andrew Gordon Wilson and Vincent Queneneville-Belair
       PyTorch 1.6 now includes Stochastic Weight Averaging
       https://pytorch.org/blog/pytorch-1.6-now-includes-stochastic-weight-averaging/

    .. [2] PyTorch - Stochastic Weight Averaging
       https://pytorch.org/docs/stable/optim.html#stochastic-weight-averaging

    Examples
    --------
    >>> optimizer, model, criterion = ...
    >>> swa_model = torch.optim.swa_utils.AveragedModel(model)
    >>> swa_start = 160
    >>> scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=300)
    >>> swa_scheduler = SWALR(optimizer, swa_lr=0.05)
    >>> swa_callback = StochasticWeightAveraging(swa_scheduler, swa_start, scheduler=scheduler)
    >>> trainer = Trainer(callbacks=[swa_callback], **kwargs)
    >>> history = trainer.fit(...)

    Now we can the SWA model by simply calling:
    >>> swa_model = swa_callback.get_swa_model()
    """

    def __init__(
        self,
        swa_scheduler: torch.optim.swa_utils.SWALR,
        start_epoch: int,
        scheduler: torch.optim.lr_scheduler._LRScheduler = None,
        metric: str = None,
        on_train: bool = True,
    ):
        super(StochasticWeightAveraging, self).__init__()

        self.swa_scheduler = swa_scheduler
        self.start_epoch = start_epoch
        self.scheduler = scheduler
        self.metric = metric
        self.on_train = on_train

        self.__swa_model = None

    def __repr__(self) -> str:
        return f"StochasticWeightAveraging(swa_scheduler={self.swa_scheduler}, start_epoch={self.start_epoch})"

    def on_fit_start(self, params_dict: dict) -> None:
        model = params_dict[ParamsDict.MODEL]
        self.__swa_model = AveragedModel(model)

    def on_train_step_end(self, params_dict: dict) -> None:
        self.__run_scheduler(params_dict=params_dict)

    def __run_scheduler(self, params_dict: dict) -> None:
        """
        Run appropiate scheduler depending on the epoch number.
        """
        epoch = params_dict[ParamsDict.EPOCH_NUMBER]
        model = params_dict[ParamsDict.MODEL]

        if epoch >= self.start_epoch:
            # run swa scheduler
            self.__swa_model.update_parameters(model)
            self.swa_scheduler.step()
        else:
            # run "standard" scheduler
            if self.metric is not None:
                key = "train" if self.on_train else "validation"
                epoch_hist = params_dict[ParamsDict.EPOCH_HISTORY]
                metric = epoch_hist[self.metric][key][-1]
                self.scheduler.step(metric)
            else:
                self.scheduler.step()

    def on_fit_end(self, params_dict: dict) -> None:
        train_loader = params_dict[ParamsDict.TRAIN_LOADER]
        torch.optim.swa_utils.update_bn(train_loader, self.__swa_model)

    def get_swa_model(self) -> torch.nn.Module:
        """
        Getter method for the Stochastic Weight Averaged model.

        Returns
        -------
        swa_model : torch.nn.Module
            SWA model.
        """
        return self.__swa_model
