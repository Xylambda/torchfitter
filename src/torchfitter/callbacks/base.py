""" Base callbacks class """
import logging

__all__ = ["Callback", "CallbackHandler"]


class Callback:
    """
    Base callbacks class.

    References
    ----------
    .. [1] Keras - keras.callbacks
       https://github.com/keras-team/keras/blob/master/keras/callbacks.py#L609
    """

    def __init__(self):
        logging.basicConfig(level=logging.INFO)
        self.callback_type = '<Trainer class>'

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

    def reset_parameters(self) -> None:
        """DEPRECATED: will be removed in future versions.
        
        Reset callback parameters when called.

        This function is called by `torchfitter.manager.Manager` class to 
        restart the initial parameters of self callback.

        It is mandatory to implement this method if one wants to use the 
        `torchfitter.manager.Manager` class along with a set of callbacks.
        """
        raise NotImplementedError(
            "Callback must implement 'reset_parameters' method in order to run"
            "multiple experiments consistently."
        )


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
        self.handle_callbacks = True

        if callbacks_list is None:
            self.handle_callbacks = False
        elif not isinstance(callbacks_list, list):
            raise TypeError("Callbacks must be a list of callbacks")

        self.callbacks_list = callbacks_list

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

    def reset_parameters(self) -> None:
        """Reset callback parameters when called.

        If callbacks `self.callbacks_list` is not empty, the called callbacks 
        must implement `reset_parameters` method.
        """
        if self.handle_callbacks:
            for callback in self.callbacks_list:
                callback.reset_parameters()
