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
>>> model = nn.Linear(in_features=1, out_features=1)
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

Now we can create the class that will handle the trainer and pass all the 
configuration we have been creating:

>>> from torchfitter.trainer import Trainer
>>> trainer = Trainer(
>>> ... model=model, 
>>> ... criterion=criterion,
>>> ... optimizer=optimizer, 
>>> ... device=device,
>>> ... callbacks=[logger, early_stopping],
>>> ... mixed_precision=True, # only works with GPU
>>> ... )

Once the trainer is created, we only need to call `fit` to optimize our model:

>>> history = trainer.fit(train_loader, val_loader, epochs=1000)

The training information you get will depend on your callbacks. After the 
optimization process ends the model is (hopefully) ready to use for inference.

`torchfitter` lets you do inference by calling the `predict` method:

>>> y_pred = trainer.predict(X_test, as_array=True)

Use `as_array=True` if you want the trainer to return a NumPy array. Notice
that you can pass a loader to the predict method too in case the Tensor is too
heavy.


Callbacks System
################

Callbacks allow interaction with the model during the fitting process. They 
provide different interaction points, at different stages in the optimization 
loop.

This callback system **was not** designed by me. It is somewhat a port from the
Keras callbacks system.

You can create your own callbacks by subclassing the Base callback and 
overriding the methods where you want to perform something.

.. code-block:: python

    import torch
    from torchfitter.conventions import ParamsDict
    from torchfitter.callbacks.base import Callback


    class ModelSaver(Callback):
        def __init__(self):
            super(ModelSaver, self).__init__()

        def __repr__(self) -> str:
            return "ModelSaver()"

        def on_epoch_end(self, params_dict):
            epoch = params_dict[ParamsDict.EPOCH_NUMBER]
            model = params_dict[ParamsDict.MODEL]
            torch.save(model.state_dict(), f"model_{epoch}.pt")

Each one of the methods receives a `params_dict` dictionary containing 
different metrics and objects that you can use to create certain logic. The 
list of objects available can be known using:

>>> from torchfitter.conventions import ParamsDict
>>> [(x, getattr(ParamsDict, x)) for x in ParamsDict.__dict__ if not x.startswith('__')]

And you can also check the doc to understand the meaning of each one of the 
parameters:

>>> from torchfitter.conventions import ParamsDict
>>> print(ParamsDict.__doc__)
