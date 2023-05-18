"""
We train a classifier with the CIFAR dataset using torchfitter to abstract the
fitting process.

Slighly modified version of these tutorials:

* PyTorch - Training a Classifier:
https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html

* Sergio Alves - Deep Learning in PyTorch with CIFAR-10 dataset:
https://medium.com/@sergioalves94/deep-learning-in-pytorch-with-cifar-10-dataset-858b504a6b54
"""

import argparse

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader, random_split
from torchfitter.callbacks import EarlyStopping, RichProgressBar
from torchfitter.conventions import ParamsDict
from torchfitter.trainer import Trainer
from torchvision.transforms import transforms


def plot_history(history):
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15, 7))
    epoch_hist = history[ParamsDict.EPOCH_HISTORY]

    ax.plot(epoch_hist[ParamsDict.LOSS]["train"], label="Train loss")
    ax.plot(
        epoch_hist[ParamsDict.LOSS]["validation"], label="Validation loss"
    )
    ax.set_title("Train and validation losses")
    ax.grid()
    ax.legend()

    plt.show()


def generate_model():
    """
    Define a convolutional neural network.
    """
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 6, 5)
            self.pool = nn.MaxPool2d(2, 2)
            self.conv2 = nn.Conv2d(6, 16, 5)
            self.fc1 = nn.Linear(16 * 5 * 5, 120)
            self.fc2 = nn.Linear(120, 84)
            self.fc3 = nn.Linear(84, 10)

        def forward(self, x):
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = torch.flatten(x, 1) # flatten all dimensions except batch
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return x

    return Net()


def generate_dataloaders(batch_size, val_size):
    """
    Generate train, validation and test dataloaders.
    """
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
    )

    # donwload data
    dataset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform
    )

    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform
    )

    # split into train and validation
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    # wrap data into datasets
    train_loader = DataLoader(
        train_ds, batch_size, shuffle=True, pin_memory=True
    )
    validation_loader = DataLoader(
        val_ds, batch_size*2, shuffle=False, pin_memory=True
    )
    test_loader = DataLoader(
        testset, batch_size*2, shuffle=False, pin_memory=True
    )

    return train_loader, validation_loader, test_loader


def main():
    # -------------------------------------------------------------------------
    # argument parsing
    parser = argparse.ArgumentParser("")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--val_size", type=int, default=5000)
    parser.add_argument("--batch_size", type=int, default=32)

    args = parser.parse_args()
    n_epochs = args.epochs
    val_size = args.val_size
    batch_size = args.batch_size

    # -------------------------------------------------------------------------
    train_loader, validation_loader, test_loader = generate_dataloaders(
        batch_size, val_size
    )
    model = generate_model()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    callbacks = [
        RichProgressBar(display_step=2),
        EarlyStopping(patience=100, load_best=True, path="cifar10.pt"),
    ]

    # -------------------------------------------------------------------------
    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        callbacks=callbacks
    )

    history = trainer.fit(
        train_loader=train_loader,
        val_loader=validation_loader,
        epochs=n_epochs
    )
    plot_history(history=history)

    # -------------------------------------------------------------------------
    _pred = trainer.predict(X=test_loader)
    y_predict = torch.argmax(_pred, dim=1)
    y_true = test_loader.dataset.targets

    print(classification_report(y_true, y_predict))


if __name__ == "__main__":
    main()