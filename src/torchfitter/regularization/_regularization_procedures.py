import torch
from typing import Union
from .base import RegularizerBase


class L1Regularization(RegularizerBase):
    """
    Applies L1 regularization over the model parameters. L1 is usually called
    'Lasso Regression' (Least Absolute Shrinkage and Selection Operator).

    Parameters
    ----------
    regularization_rate : float
        Regularization rate. Also called `lambda`.
    biases : bool, optional, default: False
        Whether to apply regularization over bias terms (True) or not (False).

    Note
    ----
    The penalty term already handles the product by the lambda regularization
    rate.
    """

    def __init__(self, regularization_rate, biases=False):
        super(L1Regularization, self).__init__(regularization_rate, biases)

    def __repr__(self):
        rpr = f"""L1Regularization(
            regularization_rate={self.rate}, biases={self.biases}
        )"""
        return rpr

    def compute_penalty(self, named_parameters, device):
        # Initialize with tensor, cannot be scalar
        penalty_term = torch.zeros(1, 1, requires_grad=True).to(device)

        for name, param in named_parameters:
            if not self.biases and name.endswith("bias"):
                pass
            else:
                penalty_term = penalty_term + param.norm(p=1)

        return self.rate * penalty_term


class L2Regularization(RegularizerBase):
    """
    Applies L2 regularization over the model parameters. L2 is usually called
    'Ridge Regression'.

    Parameters
    ----------
    regularization_rate : float
        Regularization rate. Also called `lambda`.
    biases : bool, optional, default: False
        Whether to apply regularization over bias terms (True) or not (False).

    Note
    ----
    The penalty term already handles the product by the lambda regularization
    rate.
    """

    def __init__(self, regularization_rate: float, biases: bool = False):
        super(L2Regularization, self).__init__(regularization_rate, biases)

    def __repr__(self):
        rpr = f"""L2Regularization(
            regularization_rate={self.rate}, biases={self.biases}
        )"""
        return rpr

    def compute_penalty(self, named_parameters, device: Union[str, torch.device]):
        # Initialize with tensor, cannot be scalar
        penalty_term = torch.zeros(1, 1, requires_grad=True).to(device)

        for name, param in named_parameters:
            if not self.biases and name.endswith("bias"):
                pass
            else:
                penalty_term = penalty_term + param.norm(p=2)

        return self.rate * penalty_term


class ElasticNetRegularization(RegularizerBase):
    """Linear combination of L1 and L2.

    According to [1], the lasso penalty is somewhat indifferent to the choice 
    among a set of strong but correlated variables. The ridge penalty, on the 
    other hand, tends to shrink the coefficients of correlated variables toward 
    each other. Elastic net combines both using a weighting factor:

    .. math::

        \sum_{j=1}^{p} \left( \alpha |\beta_{j}| + (1 + \alpha) \beta_{j}^{2} \right)

    Parameters
    ----------
    regularization_rate : float
        Regularization rate. Also called `lambda`.
    alpha : float
        Parameter to determine the mix of the penalties.
    biases : bool, optional, default: False
        Whether to apply regularization over bias terms (True) or not (False).

    Note
    ----
    The penalty term already handles the product by the lambda regularization
    rate.

    References
    ----------
    .. [1] Trevor Hastie, Robert Tibshirani, Jerome Friedman - The Elements of
       Statistical Learning.
    """
    def __init__(self, regularization_rate, alpha, biases=False):
        super(ElasticNetRegularization, self).__init__(
            regularization_rate, biases
        )
        self.alpha = alpha

    def __repr__(self):
        rpr = f"""ElasticNetRegularization(
            regularization_rate={self.rate},
            alpha={self.alpha},
            biases={self.biases}
        )"""
        return rpr

    def compute_penalty(self, named_parameters, device):
        # Initialize with tensor, cannot be scalar
        penalty_term = torch.zeros(1, 1, requires_grad=True).to(device)

        for name, param in named_parameters:
            if not self.biases and name.endswith("bias"):
                pass
            else:
                l1 = param.norm(p=1)
                l2 = param.norm(p=2)
                penalty_term = penalty_term + (self.alpha * l1 + (1 - self.alpha ) * l2)

        return self.rate * penalty_term
