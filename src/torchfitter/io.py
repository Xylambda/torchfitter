""" Module to handle input/output operations. """

import pickle


def save_pickle(obj, path, protocol=pickle.HIGHEST_PROTOCOL):
    """
    Save given object as pickle.

    Parameters
    ----------
    obj : object
        Object to save.
    path : str or Path
        Path where to save the pickle.
    protocol : int, optional, default: pickle.HIGHEST_PROTOCOL
        Used pickle protocol.

    """
    with open(path, 'wb') as handle:
        pickle.dump(obj, handle, protocol=protocol)


def load_pickle(path):
    """
    Load saved pickle.

    Parameters
    ----------
    path : str or Path
        Path where to save the pickle.

    Returns
    -------
    obj : object
        Loaded object.

    """
    with open(path, 'rb') as handle:
        obj = pickle.load(handle)

    return obj
