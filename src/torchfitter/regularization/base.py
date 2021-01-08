import torch


class RegularizerBase:
    """
    Base class for implementing regularization algorithms. One should inherit
    from this class the basic elements and implement his/her procedure in the
    method `_compute_penalty`.

    Parameters
    ----------
    regularization_rate : float
        Regularization rate. Also called `lambda`.
    biases : bool, optional, default: False
        Whether to apply regularization over bias terms (True) or not (False).
    """
    def __init__(self, regularization_rate, biases=False):
        self.rate = regularization_rate
        self.biases = biases

    def __call__(self, named_parameters):
        return self._compute_penalty(named_parameters)

    def _compute_penalty(self, named_parameters):
        raise NotImplementedError()