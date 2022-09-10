"""
Progress-tracking related callbacks
"""
from rich.progress import BarColumn, Progress, TimeRemainingColumn

from torchfitter.callbacks.base import Callback
from torchfitter.conventions import ParamsDict


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

    def __init__(self, update_step: int, precision: int = 2):
        super(LoggerCallback, self).__init__()
        self.update_step = update_step
        self.prec = precision

        self.set_log_name("LoggerCallback")
        self.set_log_level(20)  # info logging

    def __repr__(self) -> str:
        return (
            f"LoggerCallback(update_step={self.update_step}, "
            f"precision={self.prec})"
        )

    def on_fit_start(self, params_dict: dict) -> None:
        dev = params_dict[ParamsDict.DEVICE]
        self.logger.info(f"Starting training process on {dev}")

    def on_epoch_end(self, params_dict: dict) -> None:
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

    def on_fit_end(self, params_dict: dict) -> None:
        total_time = params_dict[ParamsDict.TOTAL_TIME]
        # final message
        self.logger.info(
            f"""End of training. Total time: {total_time:0.5f} seconds"""
        )


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

        self.set_log_name("ProgressBar")
        self.set_log_level(20)  # info logging

    def __repr__(self) -> str:
        return (
            f"RichProgressBar(display_step={self.display_step}, "
            f"log_lr={self.log_lr}, precision={self.precision})"
        )

    def on_fit_start(self, params_dict: dict) -> None:
        dev = params_dict[ParamsDict.DEVICE]
        self.logger.info(f"Starting training process on {dev}\n")

    def on_train_batch_end(self, params_dict: dict) -> None:
        epoch = params_dict[ParamsDict.EPOCH_NUMBER]
        accelerator = params_dict[ParamsDict.ACCELERATOR]

        if epoch % self.display_step == 0 or epoch == 1:
            accelerator.wait_for_everyone()
            self.progress_bar.advance(self.epoch_task, 1)

    def on_validation_batch_end(self, params_dict: dict) -> None:
        epoch = params_dict[ParamsDict.EPOCH_NUMBER]
        accelerator = params_dict[ParamsDict.ACCELERATOR]

        if epoch % self.display_step == 0 or epoch == 1:
            accelerator.wait_for_everyone()
            self.progress_bar.advance(self.epoch_task, 1)

    def on_epoch_start(self, params_dict: dict) -> None:
        # gather necessary objects
        train_loader = params_dict[ParamsDict.TRAIN_LOADER]
        val_loader = params_dict[ParamsDict.VAL_LOADER]
        epoch = params_dict[ParamsDict.EPOCH_NUMBER]
        total_epochs = params_dict[ParamsDict.TOTAL_EPOCHS]
        accelerator = params_dict[ParamsDict.ACCELERATOR]

        # compute number of batches
        n_elements = len(train_loader) + len(val_loader)

        # disable if is not the main process
        disable = not accelerator.is_local_main_process
        if epoch % self.display_step == 0 or epoch == 1:
            self.progress_bar = Progress(
                "[progress.description]{task.description}",
                "•",
                BarColumn(),
                "•",
                "[progress.percentage]{task.percentage:>3.0f}%",
                "•",
                TimeRemainingColumn(),
                disable=disable,
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
            self.logger.info(text)  # DISC: use included Rich logger?
            self.progress_bar.stop()

    def on_fit_end(self, params_dict):
        total_time = params_dict[ParamsDict.TOTAL_TIME]
        # final message
        self.logger.info(
            f"""\nEnd of training. Total time: {total_time:0.5f} seconds"""
        )

    def render_text(self, update_dict):
        text_format = ""

        for metric in update_dict:
            if metric != ParamsDict.HISTORY_LR:
                train_metric = update_dict[metric]["train"][-1]
                val_metric = update_dict[metric]["validation"][-1]

                if text_format:  # not empty
                    text_format = (
                        f"{text_format} • {metric} -> Train: "
                        f"{train_metric:.{self.prec}e} | "
                        f"Validation: {val_metric:.{self.prec}e}"
                    )
                else:
                    text_format = (
                        f"\n{metric} -> Train: "
                        f"{train_metric:.{self.prec}e} | Validation: "
                        f"{val_metric:.{self.prec}e}"
                    )
            else:
                if self.log_lr:
                    text_format = (
                        f"{text_format} • Learning Rate: "
                        f"{update_dict[metric][-1]}"
                    )

        return text_format
