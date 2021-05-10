
<p align="center">
  <img src="img/logo.png" width="320">
</p>

![GitHub tag (latest by date)](https://img.shields.io/github/v/tag/Xylambda/torchfitter?label=VERSION&style=for-the-badge)
![GitHub code size in bytes](https://img.shields.io/github/languages/code-size/Xylambda/torchfitter?style=for-the-badge)
![GitHub issues](https://img.shields.io/github/issues/Xylambda/torchfitter?style=for-the-badge)

`TorchFitter` is a simple library I created to ease the training of PyTorch
models. It features a class called `Trainer` that includes the basic 
functionality to fit models in a Keras-like style.

The library also provides a callbacks API that can be used to interact with
the model during the training process, as well as a set of basic regularization
procedures.

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

    def compute_penalty(self, named_parameters):
        # Initialize with tensor, cannot be scalar
        penalty_term = torch.zeros(1, 1, requires_grad=True)

        for name, param in named_parameters:
            if not self.biases and name.endswith("bias"):
                pass
            else:
                penalty_term = penalty_term + param.norm(p=1)

        return self.rate * penalty_term
```

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

    def on_epoch_end(self, params_dict):
        epoch = params_dict[ParamsDict.EPOCH_NUMBER]
        model = params_dict[ParamsDict.MODEL]
        torch.save(model.state_dict(), f"model_{epoch}")
```

Each method receives `params_dict`, which is a dictionary object containing the
following params:
* **training_loss** the current training loss.
* **validation_loss** the current validation loss.
* **epoch_time** the time it took to compute the current epoch.
* **epoch_number** the corresponding number of the current epoch.
* **total_epochs** the total number of epochs.
* **total_time** the total time it took to complete all epochs.
* **stop_training** whether to stop the training process (True) or not (False).
* **device** device where the model and data are stored.
* **model** the model to train.
* **history** dictionary containing:
    * **train_loss** train loss for each epoch up to the current epoch.
    * **validation_loss** validation loss for each epoch up to the current epoch.
    * **learning_rate** learning rate for each epoch up to the current epoch.

Each epoch, all dynamic params are updated according to the training process. 
You can pretty much do anything you want with those params during the training 
process.


## Custom fitting process
The current Trainer design has been created to process a dataloader that
returns 2 tensors: features and labels. Extending the Trainer class and
rewriting the methods `train_step` and `validation_step` should allow you to 
create your own custom steps as long as they receive a dataloader and they 
return the loss value as a number.

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
