""" Utilities for the training process. """
import torch
import torchmetrics
from typing import Dict, List
from torchfitter.conventions import ParamsDict


class TrainerInternalState:
    """
    Class that keeps track of the trainer internal state during the fitting 
    process.

    Parameters
    ----------
    model : torch.nn.Module
        The model to train.
    accelerator : accelerate.Accelerator
        Accelerator object.

    Attributes
    ----------
    training_loss : float
        The current training loss.
    validation_loss : float
        The current validation loss.
    epoch_time : float
        The time it took to compute the current epoch.
    epoch_number : int
        The corresponding number of the current epoch.
    total_epochs : int
        The total number of epochs.
    total_time : float
        The total time it took to complete all epochs.
    stop_training : bool
        The total time it took to complete all epochs.
    history : dict
        Dictionary containing the metrics:
        * train_loss : list of floats
            Train loss for each epoch up to the current epoch.
        * validation_loss : list of floats
            Validation loss for each epoch up to the current epoch.
        * learning_rate : list of floats
            Learning rate for each epoch up to the current epoch.

        The dictionary can be updated with more keys through the method 
        'add_history_element'.
    progress_bar : tqdm.tqdm
        Progress bar from tqdm library.
    """
    def __init__(self, model, accelerator) -> None:
        self.___initialize_dict(model=model, accelerator=accelerator)
    
    def ___initialize_dict(self, model, accelerator):
        """
        Helper function that initializes all attributes.
        """
        self.__dict__[ParamsDict.TRAIN_LOSS] = float('inf')
        self.__dict__[ParamsDict.VAL_LOSS] = float('inf')
        self.__dict__[ParamsDict.EPOCH_TIME] = 0
        self.__dict__[ParamsDict.EPOCH_NUMBER] = 1
        self.__dict__[ParamsDict.TOTAL_EPOCHS] = None
        self.__dict__[ParamsDict.TOTAL_TIME] = 0
        self.__dict__[ParamsDict.STOP_TRAINING] = False
        self.__dict__[ParamsDict.ACCELERATOR] = accelerator
        self.__dict__[ParamsDict.DEVICE] = accelerator.device
        self.__dict__[ParamsDict.MODEL] = model
        self.__dict__[ParamsDict.EPOCH_HISTORY] = {
            ParamsDict.HISTORY_TRAIN_LOSS: [],
            ParamsDict.HISTORY_VAL_LOSS: [],
            ParamsDict.HISTORY_LR: []
        }
        self.__dict__[ParamsDict.BATCH_HISTORY] = {
            ParamsDict.HISTORY_TRAIN_LOSS: [],
            ParamsDict.HISTORY_VAL_LOSS: [],
            ParamsDict.HISTORY_LR: []
        }

    def add_metrics(self, *args) -> None:
        """
        Add a metric to the history dictionary taking into account that metrics
        are computed both for training and validation steps.

        Parameters
        ----------
        *args : iterable
            Keys to add to the history dictionary.
        """
        for key in args:
            if not isinstance(key, str):
                raise TypeError(
                    f"'key' {key} must be a string, not {type(key)}"
                )
            
            for metric_type in [ParamsDict.EPOCH_HISTORY, ParamsDict.BATCH_HISTORY]:
                self.__dict__[metric_type][key] = {} # avoid key error
                self.__dict__[metric_type][key]['train'] = []
                self.__dict__[metric_type][key]['validation'] = []

    def get_single_param(self, key: str) -> object:
        """
        Retrieve a single value from the set of internal attributes of the 
        class.

        Parameters
        ----------
        key : str
            Parameter to retrieve.

        Returns
        -------
        object
            Value of passed 'key'.

        Raises
        ------
        KeyError
            If the requested attribute does not exist.
        """
        if key in self.__dict__:
            return self.__dict__[key]
        else:
            raise KeyError(f"Passed attribute '{key}' not found.")

    def get_state_dict(self) -> dict:
        """
        Return the internal dictionary that stores all parameters.

        Returns
        -------
        dict
            Internal dictionary.
        """
        return self.__dict__

    def get_params(self, *args) -> List[object]:
        """
        Parameters
        ----------
        *args : tuple
            Iterable of strings containing the keys to retrieve.

        Returns
        -------
        params : list
            List of values for each passed argument. The order is kept.

        Raises
        ------
        TypeError
            If the passed arguments are not strings.
        """
        params = []
        for key in args:
            if not isinstance(key, str):
                raise TypeError('Key to retrieve must be a string')
            
            val = self.get_single_param(key=key)
            params.append(val)

        return params

    def update_history(self, is_batch=False, **kwargs) -> None:
        """
        Update history dictionary with the passed key-value pairs.

        Parameters
        ----------
        kwargs : dict
            Dictionary with keys to update.
        """
        # select batch or epoch
        hist_type = ParamsDict.BATCH_HISTORY if is_batch else ParamsDict.EPOCH_HISTORY

        for key, value in kwargs.items():
            if key not in self.__dict__[hist_type]:
                raise KeyError(f"'{key}' not found in history dict.")
            
            self.__dict__[hist_type][key].append(value)

    def update_params(self, **kwargs) -> None:
        """
        Update internal state with the passed key-value pairs.

        Parameters
        ----------
        kwargs : dict
            Dictionary with keys to update.
        """
        for key, value in kwargs.items():
            self.__dict__[key] = value

    def update_metrics(
        self, is_train: bool, is_batch: bool=False, **kwargs
    ) -> None:
        """
        Update the metrics dictionary with the passed key-value pairs.

        Parameters
        ----------
        is_train : bool
            If True, the train metrics will be updated. Otherwise, the 
            validation metrics will be updated.
        """
        # select batch or epoch
        hist_type = ParamsDict.BATCH_HISTORY if is_batch else ParamsDict.EPOCH_HISTORY

        # select train or validation
        sub_key = 'train' if is_train else 'validation'

        for key, value in kwargs.items():
            if key not in self.__dict__[hist_type]:
                raise KeyError(f"'{key}' not found in history dict.")

            self.__dict__[hist_type][key][sub_key].append(value)
    
    def reset_parameters(self, reset_model=True):
        if reset_model:
            self.model.reset_parameters()

        self.___initialize_dict(model=self.model, device=self.device)


class MetricsHandler:
    """Class to handle the metrics computation in the process.

    The metrics must be provided using the torchmetrics package API; that is, 
    the metrics modules must support single and accumulated batch computation.
    See references for more information.

    Parameters
    ----------
    metrics_list : list of torchmetrics.Metric
        A list of all the metrics to be computed.
    metric_names : list of str
        Names of the metrics. The names are automatically computed using 
        type(metric).__name__

    References
    ----------
    .. [1] PyTorch-Lightning - torchmetrics 
       https://torchmetrics.readthedocs.io/en/latest/
    
    """
    def __init__(self, metrics_list: List[torchmetrics.Metric]) -> None:
        self.metrics_list = metrics_list

        # handle metrics if there are metrics
        self.__handle_metrics = False if self.metrics_list is None else True

        if self.__handle_metrics:
            self.metric_names = [type(metric).__name__ for metric in self.metrics_list]
        else:
            self.metric_names = None

    def single_batch_computation(
        self,
        predictions: torch.Tensor,
        target: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Perform a single batch computation on all metrics.

        Parameters
        ----------
        predictions : torch.Tensor
            Predictions of the model.
        target : torch.Tensor
            Desired output of the model.

        Returns
        -------
        results : list of torch.Tensor
            Results of each metric.
        """
        if self.__handle_metrics:
            results = {}
            for metric in self.metrics_list:
                key = type(metric).__name__
                value = metric(predictions, target).item()

                results[key] = value

            return results

    def accumulated_batch_computation(self) -> Dict[str, torch.Tensor]:
        """
        Perform the metric computation on all batches using custom 
        accumulation.

        Returns
        -------
        results : list of torch.Tensor
            Results of each metric.
        """
        if self.__handle_metrics:
            results = {}
            for metric in self.metrics_list:
                key = type(metric).__name__
                value = metric.compute().item()

                results[key] = value

            return results

    def reset_metrics(self) -> None:
        """
        Reset the internal state of the metrics in order to prepare them to 
        receive new data.
        """
        if self.__handle_metrics:
            for metric in self.metrics_list:
                metric.reset()
