""" Base callbacks class """
import logging


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

    def on_train_batch_start(self, params_dict):
        """Called at the start of a training batch.

        Subclasses should override for any actions to run. The trainer ignores
        any returned values from this function.

        Parameters
        ----------
        params_dict : dict
            Dictionary containing the parameters of the training process.
        """
        pass

    def on_train_batch_end(self, params_dict):
        """Called at the end of a training batch.

        Subclasses should override for any actions to run. The trainer ignores
        any returned values from this function.

        Parameters
        ----------
        params_dict : dict
            Dictionary containing the parameters of the training process.
        """
        pass

    def on_validation_batch_start(self, params_dict):
        """Called at the start of a validation batch.

        Subclasses should override for any actions to run. The trainer ignores
        any returned values from this function.

        Parameters
        ----------
        params_dict : dict
            Dictionary containing the parameters of the training process.
        """
        pass

    def on_validation_batch_end(self, params_dict):
        """Called at the end of a validation batch.

        Subclasses should override for any actions to run. The trainer ignores
        any returned values from this function.

        Parameters
        ----------
        params_dict : dict
            Dictionary containing the parameters of the training process.
        """
        pass

    def on_epoch_start(self, params_dict):
        """Called at the start of an epoch.

        Subclasses should override for any actions to run. The trainer ignores
        any returned values from this function.

        Parameters
        ----------
        params_dict : dict
            Dictionary containing the parameters of the training process.
        """
        pass

    def on_epoch_end(self, params_dict):
        """Called at the end of an epoch.

        Subclasses should override for any actions to run. The trainer ignores
        any returned values from this function.

        Parameters
        ----------
        params_dict : dict
            Dictionary containing the parameters of the training process.
        """
        pass

    def on_fit_start(self, params_dict):
        """Called at the start of the fitting process.

        Subclasses should override for any actions to run. The trainer ignores
        any returned values from this function.

        Parameters
        ----------
        params_dict : dict
            Dictionary containing the parameters of the training process.
        """
        pass

    def on_fit_end(self, params_dict):
        """Called at the end of the fitting process.

        Subclasses should override for any actions to run. The trainer ignores
        any returned values from this function.

        Parameters
        ----------
        params_dict : dict
            Dictionary containing the parameters of the training process.
        """
        pass


class CallbackHandler(Callback):
    """
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

    def on_train_batch_start(self, params_dict):
        if self.handle_callbacks:
            for callback in self.callbacks_list:
                callback.on_train_batch_end(params_dict)

    def on_train_batch_end(self, params_dict):
        if self.handle_callbacks:
            for callback in self.callbacks_list:
                callback.on_train_batch_end(params_dict)

    def on_validation_batch_start(self, params_dict):
        if self.handle_callbacks:
            for callback in self.callbacks_list:
                callback.on_validation_batch_start(params_dict)

    def on_validation_batch_end(self, params_dict):
        if self.handle_callbacks:
            for callback in self.callbacks_list:
                callback.on_validation_batch_end(params_dict)

    def on_epoch_start(self, params_dict):
        if self.handle_callbacks:
            for callback in self.callbacks_list:
                callback.on_epoch_start(params_dict)

    def on_epoch_end(self, params_dict):
        if self.handle_callbacks:
            for callback in self.callbacks_list:
                callback.on_epoch_end(params_dict)

    def on_fit_start(self, params_dict):
        if self.handle_callbacks:
            for callback in self.callbacks_list:
                callback.on_fit_start(params_dict)
    
    def on_fit_end(self, params_dict):
        if self.handle_callbacks:
            for callback in self.callbacks_list:
                callback.on_fit_end(params_dict)