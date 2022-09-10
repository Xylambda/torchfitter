import torch
from torch.optim.swa_utils import AveragedModel

from torchfitter.callbacks.base import Callback
from torchfitter.conventions import ParamsDict


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

    The default metric is the loss. You can choose the validation or the
    training loss or you can pass another metric by doing:

    >>> lr_sch = LearningRateScheduler(
    ...     scheduler=sch, metric='MeanSquaredError', on_train=False)
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
        return (
            f"LearningRateScheduler(scheduler={sch}, metric={self.metric}), "
            f"on_train={self.on_train}"
        )

    def on_fit_start(self, params_dict: dict) -> None:
        accelerator = params_dict[ParamsDict.ACCELERATOR]
        self.scheduler = accelerator.prepare(self.scheduler)

    def on_train_step_end(self, params_dict: dict) -> None:
        if self.metric is not None:
            key = "train" if self.on_train else "validation"
            epoch_hist = params_dict[ParamsDict.EPOCH_HISTORY]
            epoch_number = params_dict[ParamsDict.EPOCH_NUMBER]

            # avoid failing in first epoch when on_train=False
            if epoch_number > 1:
                metric = epoch_hist[self.metric][key][-1]
                self.scheduler.step(metric)
        else:
            self.scheduler.step()


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
    >>> scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    ...    optimizer, T_max=300)
    >>> swa_scheduler = SWALR(optimizer, swa_lr=0.05)
    >>> swa_callback = StochasticWeightAveraging(
    ...     swa_scheduler, swa_start, scheduler=scheduler)
    >>> trainer = Trainer(callbacks=[swa_callback], **kwargs)
    >>> history = trainer.fit(...)

    Now we can get the SWA model by simply calling:
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
        return (
            f"StochasticWeightAveraging(swa_scheduler={self.swa_scheduler}, "
            f"start_epoch={self.start_epoch})"
        )

    def on_fit_start(self, params_dict: dict) -> None:
        model = params_dict[ParamsDict.MODEL]
        accelerator = params_dict[ParamsDict.ACCELERATOR]

        self.scheduler = accelerator.prepare(self.scheduler)
        self.swa_scheduler = accelerator.prepare(self.swa_scheduler)
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
