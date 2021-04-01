""" This class wraps functionality to train PyTorch models """
import time
import torch
import logging
import warnings

from torchfitter.utils import to_device
from torchfitter.conventions import ParamsDict
from torchfitter.callbacks.base import CallbackHandler


class Trainer:
    """Trainer
    
    Class that eases the training of a PyTorch model.
    
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
    device : str, optional, default: None
        Device to perform computations. If None, the Trainer will automatically
        select the device.
    callbacks : list of torchfitter.callback.Callback
        Callbacks that allow interaction.
        
    Attributes
    ----------
    train_loss_ : list
        Training losses for each epoch.
    val_loss_ : list
        Validation losses for each epoch.
    callback_handler : torchfitter.callback.CallbackHandler
        Handles the passed callbacks.
    params_dict : dict
        Contains training params.
    
    """
    def __init__(
        self,
        model,
        criterion,
        optimizer,
        regularizer=None,
        device=None,
        callbacks=None
    ):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.regularizer=regularizer
        self.device = self._get_device(device)
        
        # send model to device
        self.model.to(self.device)

        # attributes        
        self.train_loss_ = []
        self.val_loss_ = []
        self.callback_handler = CallbackHandler(callbacks_list=callbacks)
        self.params_dict = self._initialize_params_dict()

        logging.basicConfig(level=logging.INFO)
        
    def fit(self, train_loader, val_loader, epochs):
        """Fits.
        
        Fit the model using the given loaders for the given number of epochs.
        
        Parameters
        ----------
        train_loader : torch.DataLoader
            DataLoader containing train dataset.
        val_loader : torch.DataLoader
            DataLoader containing validation dataset.
        epochs : int
            Number of training epochs.
        
        """
        # track total training time
        total_start_time = time.time()

        self.callback_handler.on_fit_begin(self.params_dict)

        # ---- train process ----
        for epoch in range(epochs):
            self._update_params_dict(
                epoch_number=epoch,
                total_epochs=epochs
            )
            self.callback_handler.on_epoch_start(self.params_dict)

            # track epoch time
            epoch_start_time = time.time()

            # train
            self.callback_handler.on_train_batch_start(self.params_dict)
            tr_loss = self._train(train_loader)
            self.callback_handler.on_train_batch_end(self.params_dict)
            self._update_params_dict(training_loss=tr_loss)

            # validation
            self.callback_handler.on_validation_batch_start(self.params_dict)
            val_loss = self._validate(val_loader)
            self.callback_handler.on_validation_batch_end(self.params_dict)
            self._update_params_dict(validation_loss=tr_loss)

            self._update_history(train_loss=tr_loss, validation_loss=val_loss)

            epoch_time = time.time() - epoch_start_time
            self._update_params_dict(epoch_time=epoch_time)

            self.callback_handler.on_epoch_end(self.params_dict)

        total_time = time.time() - total_start_time

        self._update_params_dict(total_time=total_time)
        self.callback_handler.on_fit_end(self.params_dict)
        
    def _update_params_dict(self, **kwargs):
        """

        Parameters
        ----------
        kwargs : dict
            Dictionary with keys to update.
        """
        for key, value in kwargs.items():
            self.params_dict[key] = value

    def _update_history(self, train_loss, validation_loss):
        self.params_dict['history']['train_loss'].append(train_loss)
        self.params_dict['history']['validation_loss'].append(validation_loss)

    def _initialize_params_dict(self):
        params_dict = dict(
            training_loss=0,
            validation_loss=0,
            train_batch=None,
            validation_batch=None,
            epoch_time=0,
            epoch_number=0,
            total_epochs=None,
            total_time=0,
            model_state=self.model.state_dict(),
            history=dict(
                train_loss=[],
                validation_loss=[]
            )
        )

        return params_dict
    
    def _train(self, loader):
        self.model.train()

        losses = [] # loss as mean of batch losses
        
        for features, labels in loader:
            # move to device
            features = to_device(features, self.device)
            labels = to_device(labels, self.device)
            
            # forward pass
            out = self.model(features)
            
            # loss
            loss = self._compute_loss(out, labels)
            
            # remove gradient from previous passes
            self.optimizer.zero_grad()
            
            # backprop
            loss.backward()
            
            # parameters update
            self.optimizer.step()

            losses.append(loss.item())
        
        return torch.Tensor(losses).mean().item()
    
    def _validate(self, loader):
        self.model.eval()

        losses = [] # loss as mean of batch losses
        
        with torch.no_grad():
            for features, labels in loader:
                # move to device
                features = to_device(features, self.device)
                labels = to_device(labels, self.device)
                
                out = self.model(features)
                loss = self._compute_loss(out, labels)

                losses.append(loss.item())
                
        return torch.Tensor(losses).mean().item()
    
    def _compute_loss(self, real, target):
        try:
            loss = self.criterion(real, target)
        except:
            loss = self.criterion(real, target.long())
            msg = f"Target tensor has been casted from"
            msg = f"{msg} {type(target)} to 'long' dtype to avoid errors."
            warnings.warn(msg)

        # apply regularization if any
        if self.regularizer is not None:
            penalty = self.regularizer(self.model.named_parameters())
            loss += penalty.item()
            
        return loss

    def _get_device(self, device):
        if device is None:
            dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            msg = f"Device was automatically selected: {dev}"
            warnings.warn(msg)
        else:
            dev = device

        return dev