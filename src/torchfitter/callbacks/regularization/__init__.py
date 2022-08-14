"""
Regularization related callbacks.
"""
from torch import zeros

from torchfitter.callbacks.base import Callback
from torchfitter.conventions import ParamsDict


class L1Regularization(Callback):
    """Applies L1 regularization over the model parameters.

    L1 is usually called 'Lasso Regression' (Least Absolute Shrinkage and
    Selection Operator). This callbacks is only applied to the train loss.

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
        super().__init__()

        self.rate = regularization_rate
        self.biases = biases

    def on_loss_step_end(self, params_dict: dict) -> None:
        batch_tr_loss = params_dict[ParamsDict.BATCH_TRAIN_LOSS]
        device = params_dict[ParamsDict.DEVICE]
        model = params_dict[ParamsDict.MODEL]

        # Initialize with tensor, cannot be scalar
        penalty_term = zeros(1, 1, requires_grad=True).to(device)

        for name, param in model.named_parameters():
            if not self.biases and name.endswith("bias"):
                continue

            penalty_term = penalty_term + param.norm(p=1)

        total_penalty = self.rate * penalty_term
        loss = total_penalty + batch_tr_loss

        # set loss
        params_dict[ParamsDict.BATCH_TRAIN_LOSS] = loss


class L2Regularization(Callback):
    """Applies L2 regularization over the model parameters.

    L2 is usually called 'Ridge Regression'. This callbacks is only applied to
    the train loss.

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
        super().__init__()

        self.rate = regularization_rate
        self.biases = biases

    def on_loss_step_end(self, params_dict: dict) -> None:
        batch_tr_loss = params_dict[ParamsDict.BATCH_TRAIN_LOSS]
        device = params_dict[ParamsDict.DEVICE]
        model = params_dict[ParamsDict.MODEL]

        # Initialize with tensor, cannot be scalar
        penalty_term = zeros(1, 1, requires_grad=True).to(device)

        for name, param in model.named_parameters():
            if not self.biases and name.endswith("bias"):
                continue

            penalty_term = penalty_term + param.norm(p=2)

        total_penalty = self.rate * penalty_term
        loss = total_penalty + batch_tr_loss

        # set loss
        params_dict[ParamsDict.BATCH_TRAIN_LOSS] = loss


class ElasticNetRegularization(Callback):
    r"""Linear combination of L1 and L2.

    According to [1], the lasso penalty is somewhat indifferent to the choice
    among a set of strong but correlated variables. The ridge penalty, on the
    other hand, tends to shrink the coefficients of correlated variables toward
    each other. Elastic net combines both using a weighting factor:

    .. math::

        \sum_{j=1}^{p} ( \alpha |\beta_{j}| + (1 + \alpha) \beta_{j}^{2} )

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

    def __init__(
        self, regularization_rate: float, alpha: float, biases: bool = False
    ):
        super().__init__()

        self.rate = regularization_rate
        self.biases = biases
        self.alpha = alpha

    def on_loss_step_end(self, params_dict: dict) -> None:
        batch_tr_loss = params_dict[ParamsDict.BATCH_TRAIN_LOSS]
        device = params_dict[ParamsDict.DEVICE]
        model = params_dict[ParamsDict.MODEL]

        # Initialize with tensor, cannot be scalar
        penalty_term = zeros(1, 1, requires_grad=True).to(device)

        for name, param in model.named_parameters():
            if not self.biases and name.endswith("bias"):
                continue

            l1 = param.norm(p=1)
            l2 = param.norm(p=2)
            penalty_term = penalty_term + (
                self.alpha * l1 + (1 - self.alpha) * l2
            )

        total_penalty = self.rate * penalty_term
        loss = total_penalty + batch_tr_loss

        # set loss
        params_dict[ParamsDict.BATCH_TRAIN_LOSS] = loss
