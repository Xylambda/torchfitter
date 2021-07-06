""" Callbacks for the Trainer class. """
import os
import torch
import shutil
import logging
from pathlib import Path
from .base import ManagerCallback
from torchfitter import io
from torchfitter.conventions import ManagerParamsDict


class ExperimentSaver(ManagerCallback):
    """
    Callback to save the results of a given experiment. This callback assumes 
    the use of an Early Stopping callback that saves a checkpoint in path
    'checkpoint_path'.

    By default, the ExperimentSaver stores the results in the path where the 
    manager is being runned.

    Parameters
    ----------
    checkpoint_path : str or Path
        Path where to find the Early Stopping callback checkpoint.
    folder_name : str, optional, default: 'experiment'
        Name of the folder.
    """
    def __init__(
        self, 
        checkpoint_path="checkpoint.pt", 
        folder_name='experiment'
    ) -> None:

        super(ExperimentSaver, self).__init__()

        self.folder_name = folder_name
        self.checkpoint_path = checkpoint_path

    def __repr__(self) -> str:
        return f"ExperimentSaver(checkpoint_path={self.checkpoint_path}, folder_name={self.folder_name})"

    def on_seed_experiment_end(self, params_dict: dict):
        seed = params_dict[ManagerParamsDict.CURRENT_SEED]
        model_state = params_dict[ManagerParamsDict.MODEL_STATE]
        optimizer_state = params_dict[ManagerParamsDict.OPTIMIZER_STATE]
        history = params_dict[ManagerParamsDict.HISTORY]

        self._save_experiment(
            seed=seed, 
            model_state=model_state, 
            optimizer_state=optimizer_state,
            history=history
        )

        # move saved checkpoint
        new_path = Path(f"{self.folder_name}_{seed}/best_parameters.pt")
        old_path = Path(self.checkpoint_path)
        shutil.move(old_path, new_path)

        logging.info(f'Ending training on seed {seed}')

    def _save_experiment(
        self, 
        seed: int,
        history: dict,
        model_state: torch.nn.Module, 
        optimizer_state: torch.optim.Optimizer
    ) -> None:
        """
        Helper function
        """
        # create folder
        _name = f"{self.folder_name}_{seed}"
        folder_name = Path(_name)

        if _name in os.listdir():
            pass
        else:
            os.mkdir(folder_name)

        _model_path = folder_name / 'model_parameters.pt'
        torch.save(model_state, _model_path)

        _optim_path = folder_name / 'optim_parameters.pt'
        torch.save(optimizer_state, _optim_path)

        _history_path = folder_name / 'history.pkl'
        io.save_pickle(
            history,
            _history_path
        )