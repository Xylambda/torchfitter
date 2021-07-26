""" Module that contains Manager class. """
import torch
import logging
import torchfitter
import numpy as np

from torchfitter.conventions import ManagerParamsDict, ParamsDict
from torchfitter.callbacks.base import ManagerCallbackHandler


class Manager:
    """Manager of Trainers.

    Run different experiments for the given trainer and seeds. An experiment
    will be runned for each seed. The model must to have the method 
    'reset_parameters' implemented in order to run the manager.

    Parameters
    ----------
    trainer : torchfitter.trainer.Trainer
    seeds : iterable of int
        Random seeds for the experiments. A number of experiments equal to 
        'seeds' iterable length will be runned.
    model_initial_state : dict
        Model initial state.
    optimizer_initial_state : dict
        Optimizer initial state.

    Warning
    -------
    This class has not been tested yet. Unexpected behaviour may occur.
    """
    def __init__(
        self,
        trainer: torchfitter.trainer.Trainer,
        seeds: list,
        callbacks: list=None
    ):
        self.seeds = seeds
        self.trainer = trainer
        self.callbacks_list = callbacks

        self.save_initial_states(
            model_state=self.trainer.model.state_dict(), 
            optim_state=self.trainer.optimizer.state_dict()
        )

        # attributes
        self.params_dict = self._initialize_params_dict()
        self.callback_handler = ManagerCallbackHandler(
            callbacks_list=self.callbacks_list
        )

        # set loggin level
        logging.basicConfig(level=logging.INFO)

    def _initialize_params_dict(self):
        params_dict = {
            ManagerParamsDict.SEED_LIST: self.seeds,
            ManagerParamsDict.CURRENT_SEED: None,
            ManagerParamsDict.MODEL_STATE: None,
            ManagerParamsDict.OPTIMIZER_STATE: None,
            ManagerParamsDict.HISTORY: None
        }

        return params_dict

    def _update_params_dict(self, **kwargs):
        """
        Update paramaters dictionary with the passed key-value pairs.

        Parameters
        ----------
        kwargs : dict
            Dictionary with keys to update.
        """
        for key, value in kwargs.items():
            self.params_dict[key] = value

    def run_experiments(
        self,
        train_loader: torch.utils.data.dataloader.DataLoader,
        val_loader: torch.utils.data.dataloader.DataLoader,
        epochs: int
    ) -> None:
        """Run experiments.

        Run multiple experiments for differents seeds and saves the model 
        parameters, the optimizer state and the history of an experiment.

        Parameters
        ----------
        train_loader : torch.DataLoader
            DataLoader containing train dataset.
        val_loader : torch.DataLoader
            DataLoader containing validation dataset.
        epochs : int
            Number of training epochs.
        """
        self.callback_handler.on_experiments_begin(self.params_dict)

        for seed in self.seeds:
            self._set_seed(seed)
            self._update_params_dict(
                **{ManagerParamsDict.CURRENT_SEED: seed}
            )

            # fit model
            self.callback_handler.on_seed_experiment_begin(self.params_dict)
            self.trainer.fit(train_loader, val_loader, epochs)
            self._update_params_dict(
                **{
                    ManagerParamsDict.MODEL_STATE: self.trainer.model.state_dict(),
                    ManagerParamsDict.OPTIMIZER_STATE: self.trainer.optimizer.state_dict(),
                    ManagerParamsDict.HISTORY: self.trainer.internal_state.get_state_dict()[ParamsDict.HISTORY]
                }
            )
            self.callback_handler.on_seed_experiment_end(self.params_dict)
            self.reset_parameters()

        self.callback_handler.on_experiments_end(self.params_dict)

    def reset_parameters(self) -> None:
        # reset model
        self.trainer.reset_parameters(
            reset_callbacks=True, reset_model=True
        )

        # reset optimizer
        self.trainer.optimizer.load_state_dict(
            torch.load('optimizer_initial_state.pt')
        )

    def _set_seed(self, seed: int) -> None:
        np.random.seed(seed)
        torch.manual_seed(seed)

    def save_initial_states(
        self, model_state: dict, optim_state: dict
    ) -> None:
        torch.save(model_state, 'model_initial_state.pt')
        torch.save(optim_state, 'optimizer_initial_state.pt')