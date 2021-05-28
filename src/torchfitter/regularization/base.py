""" Base class for implementing regularization procedures, """


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

    def __init__(self, regularization_rate, biases=False):
        self.rate = regularization_rate
        self.biases = biases

    def __call__(self, named_parameters, device):
        return self.compute_penalty(named_parameters, device)

    def compute_penalty(self, named_parameters, device):
        raise NotImplementedError()
