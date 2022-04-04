""" Base callbacks class """
import logging
from abc import ABC
from typing import List

from torchfitter.utils.convenience import get_logger

__all__ = ["Callback", "CallbackHandler"]


class Callback(ABC):
    """Base callbacks class.

    A callbacks allows to interact with the model along various relevant points
    during the training process. Each point is called hook, and each method of
    a callbacks allows to "attach" functionality to that particular hook.

    Attributes
    ----------
    logger : logging.Logger
        Callback logger. You can set the logging level with the
        'set_log_level'.

    References
    ----------
    .. [1] Keras - keras.callbacks
       https://github.com/keras-team/keras/blob/master/keras/callbacks.py#L609
    """

    def __init__(self):
        self.log_name: str = "Callback"
        self.logger: logging.Logger = get_logger(name=self.log_name)
        level: int = self.logger.level
        logging.basicConfig(level=level)

    def set_log_level(self, log_level) -> None:
        """
        Set the logging level this callback instance.

        Parameters
        ----------
        log_level : int
            Logging level.
        """
        self.logger.setLevel(level=log_level)
        logging.basicConfig(level=log_level)

    def on_train_step_start(self, params_dict: dict) -> None:
        """Called at the start of a training step.

        A train step will involve the processing of all train batches included
        in the train dataloader.

        Subclasses should override for any actions to run. The trainer ignores
        any returned values from this function.

        Parameters
        ----------
        params_dict : dict
            Dictionary containing the parameters of the training process.
        """
        pass

    def on_train_step_end(self, params_dict: dict) -> None:
        """Called at the end of a training step.

        A train step will involve the processing of all train batches included
        in the train dataloader.

        Subclasses should override for any actions to run. The trainer ignores
        any returned values from this function.

        Parameters
        ----------
        params_dict : dict
            Dictionary containing the parameters of the training process.
        """
        pass

    def on_train_batch_start(self, params_dict: dict) -> None:
        """Called at the start of a batch train step.

        A batch train step will involve the processing of all samples in a
        single train batch.

        Subclasses should override for any actions to run. The trainer ignores
        any returned values from this function.

        Parameters
        ----------
        params_dict : dict
            Dictionary containing the parameters of the training process.

        """
        pass

    def on_train_batch_end(self, params_dict: dict) -> None:
        """Called at the end of a batch train step.

        A batch train step will involve the processing of all samples in a
        single train batch.

        Subclasses should override for any actions to run. The trainer ignores
        any returned values from this function.

        Parameters
        ----------
        params_dict : dict
            Dictionary containing the parameters of the training process.

        """
        pass

    def on_validation_step_start(self, params_dict: dict) -> None:
        """Called at the start of a validation step.

        A validation step will involve the processing of all validation batches
        included in the validation dataloader.

        Subclasses should override for any actions to run. The trainer ignores
        any returned values from this function.

        Parameters
        ----------
        params_dict : dict
            Dictionary containing the parameters of the training process.
        """
        pass

    def on_validation_step_end(self, params_dict: dict) -> None:
        """Called at the end of a validation step.

        A validation step will involve the processing of all validation batches
        included in the validation dataloader.

        Subclasses should override for any actions to run. The trainer ignores
        any returned values from this function.

        Parameters
        ----------
        params_dict : dict
            Dictionary containing the parameters of the training process.
        """
        pass

    def on_validation_batch_start(self, params_dict: dict) -> None:
        """Called at the start of a batch validation step.

        A batch validation step will involve the processing of all samples in a
        single validation batch.

        Subclasses should override for any actions to run. The trainer ignores
        any returned values from this function.

        Parameters
        ----------
        params_dict : dict
            Dictionary containing the parameters of the training process.

        """
        pass

    def on_validation_batch_end(self, params_dict: dict) -> None:
        """Called at the end of a batch validation step.

        A batch validation step will involve the processing of all samples in a
        single validation batch.

        Subclasses should override for any actions to run. The trainer ignores
        any returned values from this function.

        Parameters
        ----------
        params_dict : dict
            Dictionary containing the parameters of the training process.

        """
        pass

    def on_epoch_start(self, params_dict: dict) -> None:
        """Called at the start of an epoch.

        Subclasses should override for any actions to run. The trainer ignores
        any returned values from this function.

        Parameters
        ----------
        params_dict : dict
            Dictionary containing the parameters of the training process.
        """
        pass

    def on_epoch_end(self, params_dict: dict) -> None:
        """Called at the end of an epoch.

        Subclasses should override for any actions to run. The trainer ignores
        any returned values from this function.

        Parameters
        ----------
        params_dict : dict
            Dictionary containing the parameters of the training process.
        """
        pass

    def on_fit_start(self, params_dict: dict) -> None:
        """Called at the start of the fitting process.

        Subclasses should override for any actions to run. The trainer ignores
        any returned values from this function.

        Parameters
        ----------
        params_dict : dict
            Dictionary containing the parameters of the training process.
        """
        pass

    def on_fit_end(self, params_dict: dict) -> None:
        """Called at the end of the fitting process.

        Subclasses should override for any actions to run. The trainer ignores
        any returned values from this function.

        Parameters
        ----------
        params_dict : dict
            Dictionary containing the parameters of the training process.
        """
        pass

    def on_loss_step_begin(self, params_dict: dict) -> None:
        """Called at the start of the loss step.

        Subclasses should override for any actions to run. The trainer ignores
        any returned values from this function.

        Parameters
        ----------
        params_dict : dict
            Dictionary containing the parameters of the training process.
        """
        pass

    def on_loss_step_end(self, params_dict: dict) -> None:
        """Called at the end of the loss step.

        Subclasses should override for any actions to run. The trainer ignores
        any returned values from this function.

        Parameters
        ----------
        params_dict : dict
            Dictionary containing the parameters of the training process.
        """
        pass


class CallbackHandler(Callback):
    """Trainer callback handler.

    Class to handle callbacks during the training process. This class is itself
    a callback.

    Parameters
    ----------
    callbacks_list : list of torchfitter.callback.Callback
        List of callbacks to handle.
    """

    def __init__(self, callbacks_list):
        self.handle_callbacks: bool = True

        if callbacks_list is None:
            self.handle_callbacks = False
        elif not isinstance(callbacks_list, list):
            raise TypeError("Callbacks must be a list of callbacks")

        self.callbacks_list: List[Callback] = callbacks_list

    def set_log_level(self, log_level: int) -> None:
        """
        Set the logging level for all callbacks contained in this instance of
        CallbacksHandler.

        Parameters
        ----------
        log_level : int
            Logging level.
        """
        if self.handle_callbacks:
            for callback in self.callbacks_list:
                callback.set_log_level(log_level=log_level)

    def on_train_step_start(self, params_dict: dict) -> None:
        """Called at the start of a training step.

        Call this method for all given callbacks list. Any returned values will
        be ignored by the trainer.

        Parameters
        ----------
        params_dict : dict
            Dictionary containing the parameters of the training process.
        """
        if self.handle_callbacks:
            for callback in self.callbacks_list:
                callback.on_train_step_start(params_dict)

    def on_train_step_end(self, params_dict: dict) -> None:
        """Called at the end of a training step.

        Call this method for all given callbacks list. Any returned values will
        be ignored by the trainer.

        Parameters
        ----------
        params_dict : dict
            Dictionary containing the parameters of the training process.
        """
        if self.handle_callbacks:
            for callback in self.callbacks_list:
                callback.on_train_step_end(params_dict)

    def on_train_batch_start(self, params_dict: dict) -> None:
        """Called at the start of a batch train step.

        A batch train step will involve the processing of all samples in a
        single train batch.

        Subclasses should override for any actions to run. The trainer ignores
        any returned values from this function.

        Parameters
        ----------
        params_dict : dict
            Dictionary containing the parameters of the training process.

        """
        if self.handle_callbacks:
            for callback in self.callbacks_list:
                callback.on_train_batch_start(params_dict)

    def on_train_batch_end(self, params_dict: dict) -> None:
        """Called at the end of a batch train step.

        A batch train step will involve the processing of all samples in a
        single train batch.

        Subclasses should override for any actions to run. The trainer ignores
        any returned values from this function.

        Parameters
        ----------
        params_dict : dict
            Dictionary containing the parameters of the training process.

        """
        if self.handle_callbacks:
            for callback in self.callbacks_list:
                callback.on_train_batch_end(params_dict)

    def on_validation_step_start(self, params_dict: dict) -> None:
        """Called at the start of a validation step.

        Call this method for all given callbacks list. Any returned values will
        be ignored by the trainer.

        Parameters
        ----------
        params_dict : dict
            Dictionary containing the parameters of the training process.
        """
        if self.handle_callbacks:
            for callback in self.callbacks_list:
                callback.on_validation_step_start(params_dict)

    def on_validation_step_end(self, params_dict: dict) -> None:
        """Called at the end of a validation step.

        Call this method for all given callbacks list. Any returned values will
        be ignored by the trainer.

        Parameters
        ----------
        params_dict : dict
            Dictionary containing the parameters of the training process.
        """
        if self.handle_callbacks:
            for callback in self.callbacks_list:
                callback.on_validation_step_end(params_dict)

    def on_validation_batch_start(self, params_dict: dict) -> None:
        """Called at the start of a batch validation step.

        A batch validation step will involve the processing of all samples in a
        single validation batch.

        Subclasses should override for any actions to run. The trainer ignores
        any returned values from this function.

        Parameters
        ----------
        params_dict : dict
            Dictionary containing the parameters of the training process.

        """
        if self.handle_callbacks:
            for callback in self.callbacks_list:
                callback.on_validation_batch_start(params_dict)

    def on_validation_batch_end(self, params_dict: dict) -> None:
        """Called at the end of a batch validation step.

        A batch validation step will involve the processing of all samples in a
        single validation batch.

        Subclasses should override for any actions to run. The trainer ignores
        any returned values from this function.

        Parameters
        ----------
        params_dict : dict
            Dictionary containing the parameters of the training process.

        """
        if self.handle_callbacks:
            for callback in self.callbacks_list:
                callback.on_validation_batch_end(params_dict)

    def on_epoch_start(self, params_dict: dict) -> None:
        """Called at the start of an epoch.

        Call this method for all given callbacks list. Any returned values will
        be ignored by the trainer.

        Parameters
        ----------
        params_dict : dict
            Dictionary containing the parameters of the training process.
        """
        if self.handle_callbacks:
            for callback in self.callbacks_list:
                callback.on_epoch_start(params_dict)

    def on_epoch_end(self, params_dict: dict) -> None:
        """Called at the end of an epoch.

        Call this method for all given callbacks list. Any returned values will
        be ignored by the trainer.

        Parameters
        ----------
        params_dict : dict
            Dictionary containing the parameters of the training process.
        """
        if self.handle_callbacks:
            for callback in self.callbacks_list:
                callback.on_epoch_end(params_dict)

    def on_fit_start(self, params_dict: dict) -> None:
        """Called at the start of the fitting process.

        Call this method for all given callbacks list. Any returned values will
        be ignored by the trainer.

        Parameters
        ----------
        params_dict : dict
            Dictionary containing the parameters of the training process.
        """
        if self.handle_callbacks:
            for callback in self.callbacks_list:
                callback.on_fit_start(params_dict)

    def on_fit_end(self, params_dict: dict) -> None:
        """Called at the end of the fitting process.

        Call this method for all given callbacks list. Any returned values will
        be ignored by the trainer.

        Parameters
        ----------
        params_dict : dict
            Dictionary containing the parameters of the training process.
        """
        if self.handle_callbacks:
            for callback in self.callbacks_list:
                callback.on_fit_end(params_dict)

    def on_loss_step_begin(self, params_dict: dict) -> None:
        """Called at the start of the loss step.

        Call this method for all given callbacks list. Any returned values will
        be ignored by the trainer.

        Parameters
        ----------
        params_dict : dict
            Dictionary containing the parameters of the training process.
        """
        if self.handle_callbacks:
            for callback in self.callbacks_list:
                callback.on_loss_step_begin(params_dict)

    def on_loss_step_end(self, params_dict: dict) -> None:
        """Called at the end of the loss step.

        Call this method for all given callbacks list. Any returned values will
        be ignored by the trainer.

        Parameters
        ----------
        params_dict : dict
            Dictionary containing the parameters of the training process.
        """
        if self.handle_callbacks:
            for callback in self.callbacks_list:
                callback.on_loss_step_end(params_dict)
