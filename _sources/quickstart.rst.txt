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

Another feature we can use is regularization. Currently, L1 and L2 are 
supported, and use them is as easy as simply create them and pass the instances
to the trainer object as we will see in a moment.

>>> from torchfitter.regularization import L1Regularization
>>> regularizer = L1Regularization(regularization_rate=0.01, biases=False)

Now we can create the class that will handle the trainer and pass all the 
configuration we have been creating:

>>> from torchfitter.trainer import Trainer
>>> trainer = Trainer(
>>> ... model=model, 
>>> ... criterion=criterion,
>>> ... optimizer=optimizer, 
>>> ... regularizer=regularizer,
>>> ... device=device,
>>> ... callbacks=[logger, early_stopping],
>>> ... mixed_precision=True, # only works with GPU
>>> ... )

Once the trainer is created, we only need to call `fit` to optimize our model:

>>> history = trainer.fit(train_loader, val_loader, epochs=1000)

The training information you get will depend on your callbacks. After the 
optimization process ends the model is ready to use.


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


Regularization
##############

The regularization system works like the callbacks system: `torchfitter` 
provides a base class that must be subclassed. Then, the method `compute_penalty`
must be filled with your algorithm. An example implementing L1

.. code-block:: python

    import torch
    from torchfitter.regularization.base import RegularizerBase


    class L1Regularization(RegularizerBase):
        def __init__(self, regularization_rate, biases=False):
            super(L1Regularization, self).__init__(regularization_rate, biases)

        def compute_penalty(self, named_parameters, device):
            # Initialize with tensor, cannot be scalar
            penalty_term = torch.zeros(1, 1, requires_grad=True).to(device)

            for name, param in named_parameters:
                if not self.biases and name.endswith("bias"):
                    pass
                else:
                    penalty_term = penalty_term + param.norm(p=1)

            return self.rate * penalty_term

Notice how the `penalty_term` is moved to the given device to avoid problems 
with tensors stored at different devices.


Running multiple experiments
############################

With `torchfitter` you can run multiple experiments sequentially for different
seeds. In order to perform various experiments, you must define an experiment
inside a function and pass it to the `Manager` class. 

The function must have 2 arguments: `seed` and `folder_name` that you can use 
to save the experiment.

Let's see an example:

.. code-block:: python

    import os
    import torch
    import numpy as np
    import torch.nn as nn
    import torch.optim as optim
    import matplotlib.pyplot as plt

    from pathlib import Path
    from torch.utils.data import DataLoader
    from sklearn.model_selection import train_test_split

    from torchfitter import io
    from torchfitter.trainer import Trainer
    from torchfitter.manager import Manager
    from torchfitter.utils import DataWrapper
    from torchfitter.regularization import L1Regularization
    from torchfitter.callbacks import (
        LoggerCallback, 
        EarlyStopping, 
        LearningRateScheduler
    )

    DATA_PATH = <path_to_data>

    # define experiment function
    def experiment_func(seed, folder_name):
        subfolder = folder_name / f"experiment_{seed}"
        
        if f"experiment_{seed}" not in os.listdir(folder_name):
            os.mkdir(subfolder)
        
        # ---------------------------------------------------------------------
        # split
        X = np.load(DATA_PATH / "features.npy")
        y = np.load(DATA_PATH / "labels.npy")
        y = y.reshape(-1,1)
        
        # simplest case of cross-validation
        X_train, X_val, y_train, y_val = train_test_split(
            X, 
            y, 
            test_size=0.33, 
            random_state=42
        )
        
        # wrap data in Dataset
        train_wrapper = DataWrapper(
            X_train, 
            y_train, 
            dtype_X='float', 
            dtype_y='float'
        )
        val_wrapper = DataWrapper(
            X_val,
            y_val,
            dtype_X='float',
            dtype_y='float'
        )

        # torch Loaders
        train_loader = DataLoader(train_wrapper, batch_size=64, pin_memory=True)
        val_loader = DataLoader(val_wrapper, batch_size=64, pin_memory=True)

        # ---------------------------------------------------------------------
        # model creatiom
        model = nn.Linear(in_features=1, out_features=1)
        
        # optimization settings 
        regularizer = L1Regularization(regularization_rate=0.01, biases=False)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.005)
        
        # ---------------------------------------------------------------------
        callbacks = [
            LoggerCallback(update_step=100),
            EarlyStopping(patience=50, load_best=False, path=subfolder / 'checkpoint.pt'),
            LearningRateScheduler(
                scheduler=optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.9)
            )
        ]

        # trainer
        trainer = Trainer(
            model=model, 
            criterion=criterion,
            optimizer=optimizer, 
            regularizer=regularizer,
            callbacks=callbacks,
        )
        
        # run training
        history = trainer.fit(train_loader, val_loader, 5000, disable_pbar=True)
        
        # ---------------------------------------------------------------------
        # model state
        torch.save(trainer.model.state_dict(), subfolder / 'model_state.pt')
        
        # optim state
        torch.save(trainer.optimizer.state_dict(), subfolder / 'optim_state.pt')
        
        # history
        io.save_pickle(
            obj=history,
            path=subfolder / 'history.pkl'
        )

    # define random seeds
    seeds = (0, 5, 10)
    folder = Path('experiments')

    manager = Manager(
        seeds=seeds,
        folder_name=folder
    )
    # run experiments
    manager.run_experiments(experiment_func=experiment_func)