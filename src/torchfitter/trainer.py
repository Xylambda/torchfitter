""" This class wraps functionality to train PyTorch models """
import time
import torch
import logging
import warnings


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
    logger_kwards : dict
        Args for logger. Currently, only 'show' and 'update_step' are available
        parameters.
    regularizer : torchfitter.regularizer, optional, default: None
        Procedure to apply penalties to the loss function.
    device : str, optional, default: None
        Device to perform computations. If None, the Trainer will automatically
        select the device.
        
    Attributes
    ----------
    train_loss_ : list
        Training losses for each epoch.
    val_loss_ : list
        Validation losses for each epoch.
    
    """
    def __init__(
        self,
        model,
        criterion,
        optimizer,
        logger_kwargs,
        regularizer=None,
        device=None
    ):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.regularizer=regularizer
        self.logger_kwargs = logger_kwargs
        self.device = self._get_device(device)
        
        # send model to device
        self.model.to(self.device)

        # attributes        
        self.train_loss_ = []
        self.val_loss_ = []

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

        # ---- train process ----
        for epoch in range(epochs):
            # track epoch time
            epoch_start_time = time.time()

            # train
            tr_loss = self._train(train_loader)
            
            # validate
            val_loss = self._validate(val_loader)
            
            self.train_loss_.append(tr_loss)
            self.val_loss_.append(val_loss)

            epoch_time = time.time() - epoch_start_time
            self._logger(
                tr_loss, 
                val_loss, 
                epoch+1, 
                epochs, 
                epoch_time, 
                **self.logger_kwargs
            )

        total_time = time.time() - total_start_time

        # final message
        logging.info(
            f"""End of training. Total time: {round(total_time, 5)} seconds"""
        )
        
    
    def _logger(
        self, 
        tr_loss, 
        val_loss, 
        epoch, 
        epochs, 
        epoch_time, 
        show=True, 
        update_step=20
    ):
        if show:
            if epoch % update_step == 0 or epoch == 1:
                # to satisfy pep8 common limit of characters
                msg = f"Epoch {epoch}/{epochs} | Train loss: {tr_loss}" 
                msg = f"{msg} | Validation loss: {val_loss}"
                msg = f"{msg} | Time/epoch: {round(epoch_time, 5)} seconds"

                logging.info(msg)
    
    def _train(self, loader):
        self.model.train()
        
        for features, labels in loader:
            # move to device
            features, labels = self._to_device(features, labels, self.device)
            
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
            
        return loss.item()
    
    def _to_device(self, features, labels, device):
        return features.to(device), labels.to(device)
    
    def _validate(self, loader):
        self.model.eval()
        
        with torch.no_grad():
            for features, labels in loader:
                # move to device
                features, labels = self._to_device(
                    features, 
                    labels, 
                    self.device
                )
                
                out = self.model(features)
                loss = self._compute_loss(out, labels)
                
        return loss.item()
    
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