
<p align="center">
  <img src="img/logo.png" width="650">
</p>

![GitHub tag (latest by date)](https://img.shields.io/github/v/tag/Xylambda/torchfitter?label=VERSION&style=badge)
![GitHub code size in bytes](https://img.shields.io/github/languages/code-size/Xylambda/torchfitter?style=badge)
![GitHub issues](https://img.shields.io/github/issues/Xylambda/torchfitter?style=badge)
![workflow](https://github.com/Xylambda/torchfitter/actions/workflows/cicd.yml/badge.svg)
[![doc](https://img.shields.io/badge/DOCS-documentation-blue.svg?style=badge)](https://xylambda.github.io/torchfitter/)

`torchfitter` is a simple library to ease the training of PyTorch models. It 
features a class called `Trainer` that includes the basic functionality to fit 
models in a Keras-like style.

Internally, `torchfitter` leverages the power of [accelerate](https://huggingface.co/docs/accelerate/)
to handle the device management.

The library also provides a callbacks API that can be used to interact with
the model during the training process, as well as a set of basic regularization
procedures.

Additionally, you will find the `Manager` class which allows you to run 
multiple experiments for different random seeds.

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

## Features

|                          | Supported | Not supported | Planned |
|--------------------------|-----------|---------------|---------|
|      Basic training loop |     x     |               |         |
|        Gradient Clipping |     x     |               |         |
|    Gradient Accumulation |     x     |               |         |
|     Multi-device support |     x     |               |         |
|           Regularization |     x     |               |         |
|  In-loop metrics support |     x     |               |         |
| Mixed precision training |     x     |               |         |
|         Callbacks System |     x     |               |         |
|    Hyperparameter search |           |       x       |         |
|            Warm Training |           |       x       |    x    |

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

model = nn.Linear(in_features=1, out_features=1)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters())
regularizer = L1Regularization(regularization_rate=0.01, biases=False)

# callbacks
logger = LoggerCallback(update_step=50)
early_stopping = EarlyStopping(patience=50, load_best=True, path='checkpoint.pt')
scheduler = LearningRateScheduler(
    scheduler=optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.9)
)

trainer = Trainer(
    model=model, 
    criterion=criterion,
    optimizer=optimizer, 
    regularizer=regularizer,
    mixed_precision=True,
    accumulate_iter=4, # accumulate gradient every 4 iterations,
    gradient_clipping='norm',
    gradient_clipping_kwrgs={'max_norm': 1.0, 'norm_type': 2.0},
    callbacks=[logger, early_stopping, scheduler]
)

trainer.fit(train_loader, val_loader, epochs=1000)
```

Since `torchfitter` leverages the power of `accelerate`, the device management
will rely on the latter. You can pass your own `accelerate.Accelerator` 
object to fine tune its parameters:

```python
from accelerate import Accelerator
from torchfitter.trainer import Trainer


accelerator = Accelerator(...)
trainer = Trainer(
    **kwargs,
    accelerator=accelerator
)
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
in order to avoid operations with tensors stored at different devices.

## Callbacks
Callbacks allow you to interact with the model during the fitting process. They
provide with different methods that are called at different stages. To create a 
callback simply extend the base class and fill the desired methods.

```python
import torch
from torchfitter.conventions import ParamsDict
from torchfitter.callbacks.base import Callback


class ModelCheckpoint(Callback):
    def __init__(self):
        super(ModelCheckpoint, self).__init__()

    def __repr__(self) -> str:
        return "ModelCheckpoint()"

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
```

And you can also check the doc to understand the meaning of each one of the 
parameters:
```python
>>> from torchfitter.conventions import ParamsDict
>>> print(ParamsDict.__doc__)
```

`NOTE:` the callbacks design can be considered as a port from Keras design. 
`I AM NOT` the author of this callback sysem design despite the fact that I 
made some minor design changes. Find more in the `Credits` section.


## FAQ
* **Do you know Pytorch-Lightning/FastAI?**

I know them and I think **they are awesome**. This is a personal project though
I must say the trainer is reasonably well-equiped.

* **Why is the `validation loader` not optional?**

Because I think it enforces good ML practices that way.

* **Why didn't you implement the optimization steps in the model object?**

It is certainly another approach you may take when building an optimization 
loop (PyTorch-Lightning works this way), but I don't like my abstract data 
types to track way too many things in addition to being torch.nn.Module types. 
Functionality should be **clear and atomic**: the model tracks gradients and 
the trainer cares about the optimization process.

* **I have a suggestion/question**

Thank you! Do not hesitate to open an issue and I'll do my best to answer you.

## CREDITS

* [Keras API](https://keras.io/api/).

* [torchmetrics](https://torchmetrics.readthedocs.io/en/latest/)

* [fastai](https://docs.fast.ai/)


## Cite
If you've used this library for your projects please cite it:

```latex
@misc{alejandro2019torchfitter,
  title={torchfitter - Simple Trainer to Optimize PyTorch Models},
  author={Alejandro Pérez-Sanjuán},
  year={2020},
  howpublished={\url{https://github.com/Xylambda/torchfitter}},
}
```