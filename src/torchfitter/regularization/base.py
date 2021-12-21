""" Base class for implementing regularization procedures, """
import torch
from typing import Generator


class RegularizerBase:
    """
    Base class for implementing regularization algorithms. One should inherit
    from this class the basic elements and implement his/her procedure in the
    method `compute_penalty`.

    Parameters
    ----------
    regularization_rate : float
        Regularization rate. Also called `lambda`.
    biases : bool, optional, default: False
        Whether to apply regularization over bias terms (True) or not (False).
    """

    def __init__(self, regularization_rate: float, biases: bool = False):
        self.rate = regularization_rate
        self.biases = biases

    def __repr__(self):
        rpr = f"""RegularizerBase(
            regularization_rate={self.rate},
            biases={self.biases}
        )"""
        return rpr

    def __call__(
        self,
        named_parameters: Generator[str, torch.Tensor, None],
        device: torch.device,
    ) -> torch.Tensor:
        return self.compute_penalty(named_parameters, device)

    def compute_penalty(
        self,
        named_parameters: Generator[str, torch.Tensor, None],
        device: torch.device,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        named_parameters : generator
            Named parameters generator from a torch.nn.Module.
        devide : torch.device
            Device where to compute the regularization.
        """
        raise NotImplementedError()
