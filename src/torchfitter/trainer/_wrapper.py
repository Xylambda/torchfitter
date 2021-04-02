""" Model wrapper. """


class ModelWrapper:
    """
    Thin class to wrap a torch.nn.Module and add some functionality.

    Parameters
    ----------
    model : torch.nn.Module
        Model to train.
    device : str
        Device to move the model to.
    """
    def __init__(self, model, device):
        super(ModelWrapper, self).__init__()

        self.model = model.to(device)
        self.stop_training = False

    def __call__(self, *input, **kwargs):
        return self.model(*input, **kwargs)

    def train(self, *input, **kwargs):
        self.model.train(*input, **kwargs)

    def eval(self, *input, **kwargs):
        self.model.eval(*input, **kwargs)

    def state_dict(self, *input, **kwargs):
        return self.model.state_dict(*input, **kwargs)

    def load_state_dict(self, *input, **kwargs):
        self.model.load_state_dict(*input, **kwargs)

    def named_parameters(self, *input, **kwargs):
        return self.model.named_parameters(*input, **kwargs)

    def parameters(self, *input, **kwargs):
        return self.model.parameters(*input, **kwargs)