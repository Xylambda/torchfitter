""" Module that contains Manager class. """
import os
import torch
import random
import numpy as np
from typing import Callable, Iterable


class Manager:
    """
    This class runs multiple experiments (one for each seed) sequentially.

    Parameters
    ----------
    seeds : iterable of ints
        Seeds to use in the experiments.
    """

    def __init__(
        self,
        seeds: Iterable[int],
        folder_name: str,
        deterministic: bool = False,
    ):
        self.seeds = seeds
        self.folder_name = folder_name
        self.deterministic = deterministic

        if str(self.folder_name) not in os.listdir():
            os.mkdir(self.folder_name)

    def run_experiments(self, experiment_func: Callable) -> None:
        """Run 'experiment_func' for each random seed.

        All the logic, including model and optimizer states saving, must be
        coded in the 'experiment_func' variable.

        'experiment_func' must receive 2 arguments: 'seed' and 'folder_name'.
        These arguments may be used by the user to create the necessary folders
        to save the experiment.

        Parameters
        ----------
        experiment_func : function
            Experiment function.
        """
        for seed in self.seeds:
            self.__set_seed(seed=seed)
            experiment_func(seed=seed, folder_name=self.folder_name)

    def __set_seed(self, seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.use_deterministic_algorithms(self.deterministic)
