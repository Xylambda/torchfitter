
<p align="center">
  <img src="img/logo.png" width="400">
</p>

![GitHub tag (latest by date)](https://img.shields.io/github/v/tag/Xylambda/torchfitter?label=VERSION&style=for-the-badge)
![GitHub code size in bytes](https://img.shields.io/github/languages/code-size/Xylambda/torchfitter?style=for-the-badge)
![GitHub last commit](https://img.shields.io/github/last-commit/Xylambda/torchfitter?style=for-the-badge)
![GitHub issues](https://img.shields.io/github/issues/Xylambda/torchfitter?style=for-the-badge)

`TorchFitter` is a simple library I created to ease the training of PyTorch
models. It features a class called `Trainer` that includes the basic 
functionality to fit models in a Keras-like style.

## Installation
Normal user:
```bash
git clone https://github.com/Xylambda/torchfitter.git
pip install torchfitter/.
```

alternatively:
```bash
git clone https://github.com/Xylambda/torchfitter.git
pip install torchfitter/. -r torchfitter/requirements-base.txt
```

Developer:
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
you need/want to do it.
```python
import torch.nn as nn
import torch.optim as optim
from torchfitter.trainer import Trainer
from torchfitter.regularization import L1Regularization


# get device
device = "cuda" if torch.cuda.is_available() else "cpu"

model = nn.Linear(in_features=1, out_features=1)
model.to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters())
regularizer = L1Regularization(regularization_rate=0.01, biases=False)

trainer = Trainer(
    model=model, 
    criterion=criterion,
    optimizer=optimizer, 
    logger_kwargs={'show': True, 'update_step':20},
    regularizer=regularizer,
    device=device
)

trainer.fit(train_loader, val_loader, epochs=10)
```

A logger will keep you up to date about the training process.
After the process ends, you can access the validation and train losses:
```python
train_loss = trainer.train_loss_
val_loss = trainer.val_loss_
```

## About regularization
`TorchFitter` includes regularization algorithms but you can also create your
own procedures. To create your own algorithms you just:
1. Inherit from `RegularizerBase` and call the `super` operator appropiately.
2. Implement the procedure in the `_compute_penalty` method.

Here's an example implementing L1 from scratch:

```python
import torch
from torchfitter.base import RegularizerBase


class L1Regularization(RegularizerBase):
    def __init__(self, regularization_rate, biases=False):
        super(L1Regularization, self).__init__(regularization_rate, biases)

    def _compute_penalty(self, named_parameters):
        # Initialize with tensor, cannot be scalar
        penalty_term = torch.zeros(1, 1, requires_grad=True)

        for name, param in named_parameters:
            if not self.biases and name.endswith("bias"):
                pass
            else:
                penalty_term = penalty_term + param.norm(p=1)

        return self.rate * penalty_term
```

## FAQ
* **Do you know Pytorch-Lightning?**

I know it and I think is awesome; but I like to build and test my own stuff.

* **Why is the `validation loader` not optional?**

Because I think it enforces good ML practices that way.

* **I have a suggestion/question**

Thank you! Do not hesitate to open an issue and I'll do my best to answer you.

## TODO
* Add support for EarlyStopping.
* Add support for computing metrics in the training loop.
* Improve `logger_kwargs` management.

## CREDITS
<div>Icons made by <a href="https://www.flaticon.com/authors/vignesh-oviyan" 
title="Vignesh Oviyan">Vignesh Oviyan</a> from <a href="https://www.flaticon.com/" 
title="Flaticon">www.flaticon.com</a></div>
