===========
Quick Start
===========

Welcome to `torchfitter`, a light class that allows to optimize PyTorch models 
in a Keras-like style.

In this guide we will cover the basics of the library.

Training Basics
###############

The first thing to know about `torchfitter` is that it tries to follow the 
PyTorch conventions for loading data and optimizing models. That means that we
need to start by having 2 loaders: the validation and the train loader.

This guide assumes basic familiarity with the PyTorch API, so we will not cover 
that here.

>>> from torch.utils.data import DataLoader
>>> train_loader = DataLoader(...)
>>> val_loader = DataLoader(...)

After that, we can create a model and declare an optimizer and a criterion as 
one would normally do in PyTorch.

>>> import torch.nn as nn
>>> import torch.optim as optim
>>> device = "cuda" if torch.cuda.is_available() else "cpu"
>>> model = nn.Linear(in_features=1, out_features=1)
>>> model.to(device) # do this before declaring the optimizer
>>> criterion = nn.MSELoss()
>>> optimizer = optim.Adam(model.parameters())

Everything should be familiar until this point, where we can start using 
`torchfitter`.

Let's start by creating 2 callbacks: an early stopping and a logger. 
`torchfitter` includes a set of callbacks by default, but you can create your 
own callbacks (we will cover this later).

>>> from torchfitter.callbacks import LoggerCallback, EarlyStopping
>>> logger = LoggerCallback(update_step=50)
>>> early_stopping = EarlyStopping(patience=50, load_best=True, path='checkpoint.pt')

Another feature we can use is regularization. Currently, L1 and L2 are 
supported, and use them is as easy as simply create them and pass the instances
to the trainer object as we will see in a moment.

>>> from torchfitter.regularization import L1Regularization
>>> regularizer = L1Regularization(regularization_rate=0.01, biases=False)

Now we can create the class that will handle the trainer and pass all the 
configuration we have been creating:

>>> from torchfitter.trainer import Trainer
>>> trainer = Trainer(
... model=model, 
... criterion=criterion,
... optimizer=optimizer, 
... regularizer=regularizer,
... device=device,
... callbacks=[logger, early_stopping, scheduler]
... )

Make sure to pass the device since `torchfitter` will grab whatever is 
available internally, which can cause problems when processing tensors from 
different devices.

Once the trainer is created, we only need to call `fit` to optimize our model:

>>> trainer.fit(train_loader, val_loader, epochs=1000)

The training information you get will depend on your callbacks. A progress bar 
will also be displayed unless `disable_pbar` is set to `True` in the `fit` 
method.

After the optimization process ends, the model is now ready to use.


Callbacks System
################

Callbacks allow interaction with the model during the fitting process. They 
provide different interaction points, at different stages in the optimization 
loop.

This callback system **was not** designed by me. It is somewhat a port from the
Keras callbacks system.