""" Callbacks for the manager class """
import subprocess
from typing import List

import torch

from torchfitter.callbacks.base import Callback
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
    path : str or Path, optional, default: 'checkpoint.pt'
        Path to store the best observed parameters.
    """

    def __init__(self, patience=50, load_best=True, path="checkpoint.pt"):
        super(EarlyStopping, self).__init__()
        self.path = path
        self.patience = patience
        self.load_best = load_best

        self.set_log_name("EarlyStopping")
        self.set_log_level(20)  # info logging

    def __repr__(self) -> str:
        return (
            f"EarlyStopping(patience={self.patience}, "
            f"load_best={self.load_best})"
        )

    def on_fit_start(self, params_dict: dict) -> None:
        self.wait = 0
        self.stopped_epoch = 0
        self.best = float("inf")

    def on_epoch_end(self, params_dict: dict) -> None:
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

    def on_fit_end(self, params_dict: dict) -> None:
        if self.stopped_epoch > 0:
            self.logger.info(
                f"Early stopping applied at epoch: {self.stopped_epoch}"
            )


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

        self.set_log_name("GPU Stats")
        self.set_log_level(20)  # info logging

    def __repr__(self) -> str:
        return (
            f"GPUStats(format={self.format}, queries={self.queries}, "
            f"queries={self.queries})"
        )

    def on_epoch_end(self, params_dict: dict) -> None:
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
