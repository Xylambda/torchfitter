
<p align="center">
  <img src="img/logo.png" width="320">
</p>

![GitHub tag (latest by date)](https://img.shields.io/github/v/tag/Xylambda/torchfitter?label=VERSION&style=for-the-badge)
![GitHub code size in bytes](https://img.shields.io/github/languages/code-size/Xylambda/torchfitter?style=for-the-badge)
![GitHub issues](https://img.shields.io/github/issues/Xylambda/torchfitter?style=for-the-badge)

`torchfitter` is a simple library I created to ease the training of PyTorch
models. It features a class called `Trainer` that includes the basic 
functionality to fit models in a Keras-like style.

The library also provides a callbacks API that can be used to interact with
the model during the training process, as well as a set of basic regularization
procedures.

Additionally, you will find the `Manager` class which allows you to run 
multiple experiments for different random seeds. The class is still in testing
stage, so unexpected behaviour may occur when using it.

## Installation
**Normal user**
```bash
git clone https://github.com/Xylambda/torchfitter.git
pip install torchfitter/.
```

**Developer**
```bash
git clone https://github.com/Xylambda/torchfitter.git
pip install -e torchfitter/. -r torchfitter/requirements-dev.txt
```

## Tests
To run the tests you must install the library as a `developer`.
```bash
cd torchfitter/
pytest -v tests/
```

## Usage
Assume we already have `DataLoaders` for the train and validation sets. 
```python
from torch.utils.data import DataLoader


train_loader = DataLoader(...)
val_loader = DataLoader(...)
```

Then, create the optimizer and the loss criterion as usual. Pass them to the
trainer along the PyTorch model. You can also add a regularization procedure if 
you need/want to do it. The same goes for callbacks: create the desired
callbacks and pass them to the trainer as a list.
```python
import torch.nn as nn
import torch.optim as optim
from torchfitter.trainer import Trainer
from torchfitter.regularization import L1Regularization
from torchfitter.callbacks import (
    LoggerCallback,
    EarlyStopping,
    LearningRateScheduler
)


# get device
device = "cuda" if torch.cuda.is_available() else "cpu"

model = nn.Linear(in_features=1, out_features=1)
model.to(device) # do this before declaring the optimizer

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters())
regularizer = L1Regularization(regularization_rate=0.01, biases=False)

# callbacks
logger = LoggerCallback(update_step=50)
early_stopping = EarlyStopping(patience=50, load_best=True)
scheduler = LearningRateScheduler(
    scheduler=optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.9)
)

trainer = Trainer(
    model=model, 
    criterion=criterion,
    optimizer=optimizer, 
    regularizer=regularizer,
    device=device,
    callbacks=[logger, early_stopping, scheduler]
)

trainer.fit(train_loader, val_loader, epochs=1000)
```


## Regularization
`TorchFitter` includes regularization algorithms but you can also create your
own procedures. To create your own algorithms you just:
1. Inherit from `RegularizerBase` and call the `super` operator appropiately.
2. Implement the procedure in the `compute_penalty` method.

Here's an example implementing L1 from scratch:

```python
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
```

Notice how the `penalty_term` is moved to the given `device`. This is necessary
in order to avoid operations with tensors stored in different devices.

## Callbacks
Callbacks allow you to interact with the model during the fitting process. They
provide with different methods at different stages. To create a callback simply 
extend the base class and fill the desired methods.

```python
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
```

Each method receives `params_dict`, which is a dictionary object containing the
internal training parameters. You can see the pair key value of each parameter
of the conventions:

```python
>>> from torchfitter.conventions import ParamsDict
>>> [(x, getattr(ParamsDict, x)) for x in ParamsDict.__dict__ if not x.startswith('__')]
[('TRAIN_LOSS', 'training_loss'),
 ('VAL_LOSS', 'validation_loss'),
 ('EPOCH_TIME', 'epoch_time'),
 ('EPOCH_NUMBER', 'epoch_number'),
 ('TOTAL_EPOCHS', 'total_epochs'),
 ('TOTAL_TIME', 'total_time'),
 ('STOP_TRAINING', 'stop_training'),
 ('DEVICE', 'device'),
 ('MODEL', 'model'),
 ('HISTORY', 'history'),
 ('HISTORY_TRAIN_LOSS', 'train_loss'),
 ('HISTORY_VAL_LOSS', 'validation_loss'),
 ('HISTORY_LR', 'learning_rate'),
 ('PROG_BAR', 'progress_bar')]
```

And you can also check the doc to understand the meaning of each one of the 
parameters:
```python
>>> from torchfitter.conventions import ParamsDict
>>> print(ParamsDict.__doc__)

    Naming conventions for torchfitter.trainer.Trainer internal parameters.

    Attributes
    ----------
    TRAIN_LOSS : str
        The current training loss.
    VAL_LOSS : str
        The current validation loss.
    EPOCH_TIME : str
        The time it took to compute the current epoch.
    EPOCH_NUMBER : str
        The corresponding number of the current epoch.
    TOTAL_EPOCHS : str
        The total number of epochs.
    TOTAL_TIME : str
        The total time it took to complete all epochs.
    STOP_TRAINING : str
        The total time it took to complete all epochs.
    DEVICE : str
        Device where the model and data are stored.
    MODEL : str
        The model to train.
    HISTORY : str
        Dictionary containing the metrics:
        * ParamsDict.HISTORY_TRAIN_LOSS
        * ParamsDict.HISTORY_VAL_LOSS
        * ParamsDict.HISTORY_LR
    HISTORY_TRAIN_LOSS : str
        Train loss for each epoch up to the current epoch.
    HISTORY_VAL_LOSS : str
        Validation loss for each epoch up to the current epoch.
    HISTORY_LR : str
        Learning rate for each epoch up to the current epoch.
    PROG_BAR : str
        Progress bar from tqdm library.
```

`NOTE:` the callbacks design can be considered as a port from Keras design. 
`I AM NOT` the author of this callbacks design despite the fact that I made 
some minor design changes. Find more in the `Credits` section.

## Custom fitting process
The current Trainer design has been created to process a dataloader that
returns 2 tensors: features and labels. Extending the Trainer class and
rewriting the methods `train_step` and `validation_step` should allow you to 
create your own custom steps as long as they receive a dataloader and they 
return the loss value as a number.

Additionally, the loss computation can also be customized; just remember to
handle the regularization if any.

```python
from torchfitter.trainer import Trainer


class MyTrainer(Trainer):
    def __init__(self, **kwargs):
        super(MyTrainer, self).__init__(**kwargs)

    def train_step(self, loader):
        # ...
        return loss # must be a number

    def validation_step(self, loader):
        # ...
        return loss # must be a number

    def compute_loss(self, real, target):
        # ...
        return loss # loss graph
```

## FAQ
* **Do you know Pytorch-Lightning?**

I know it and I think **it is awesome**. This is a personal project. I wouln't
hesitate to use [Pytorch-Lightning](https://www.pytorchlightning.ai/) for 
more complex tasks, like distributed training on multiple GPUs.

* **Why is the `validation loader` not optional?**

Because I think it enforces good ML practices that way.

* **I have a suggestion/question**

Thank you! Do not hesitate to open an issue and I'll do my best to answer you.

## CREDITS
* <div>Icons made by <a href="https://www.flaticon.com/authors/vignesh-oviyan" title="Vignesh Oviyan">Vignesh Oviyan</a> from <a href="https://www.flaticon.com/" title="Flaticon">www.flaticon.com</a></div>

* [Keras API](https://keras.io/api/).
