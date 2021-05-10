import torch
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
            regularization_rate={self.rate},
            biases={self.biases}
        )"""
        return rpr

    def compute_penalty(self, named_parameters):
        # Initialize with tensor, cannot be scalar
        penalty_term = torch.zeros(1, 1, requires_grad=True)

        for name, param in named_parameters:
            if not self.biases and name.endswith("bias"):
                pass
            else:
                penalty_term = penalty_term + param.norm(p=1)

        return self.rate * penalty_term


class L2Regularization(RegularizerBase):
    """
    Applies L2 regularization over the model parameters. L2 is usually called
    'Lasso Regression'.

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
        super(L2Regularization, self).__init__(regularization_rate, biases)

    def __repr__(self):
        rpr = f"""L2Regularization(
            regularization_rate={self.rate},
            biases={self.biases}
        )"""
        return rpr

    def compute_penalty(self, named_parameters):
        # Initialize with tensor, cannot be scalar
        penalty_term = torch.zeros(1, 1, requires_grad=True)

        for name, param in named_parameters:
            if not self.biases and name.endswith("bias"):
                pass
            else:
                penalty_term = penalty_term + param.norm(p=2)

        return self.rate * penalty_term


class ElasticNetRegularization(RegularizerBase):
    """
    Implements Elastic Net regression algorithm for regression. ElasticNet
    simply applies a combination of L1 and L2:
        ElasticNet = l1_lambda * L1 + l2_lambda * L2

    Parameters
    ----------
    l1_lambda : float
        Lambda value for L1 (lasso).
    l2_lambda : float
        Lambda value for L2 (ridge).
    biases : bool, optional, default: False
        Whether to apply regularization over bias terms (True) or not (False).
    """

    def __init__(self, l1_lambda, l2_lambda, biases=False):
        super(ElasticNetRegularization, self).__init__(
            regularization_rate=None, biases=biases
        )

        self.l1_lambda = l1_lambda
        self.l2_lambda = l2_lambda

    def __repr__(self):
        rpr = f"""ElasticNetRegularization(
            l1_lambda={self.l1_lambda},
            l2_lambda={self.l2_lambda},
            biases={self.biases}
        )"""
        return rpr

    def compute_penalty(self, named_parameters):
        # Initialize with tensor, cannot be scalar
        penalty_term = torch.zeros(1, 1, requires_grad=True)

        for name, param in named_parameters:
            if not self.biases and name.endswith("bias"):
                pass
            else:
                l1 = self.l1_lambda * param.norm(p=1)
                l2 = self.l2_lambda * param.norm(p=2)

                penalty_term = l1 + l2

        return penalty_term
