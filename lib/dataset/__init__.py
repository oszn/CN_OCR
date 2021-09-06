from ._360cc import _360CC
from ._own import _OWN
from ._fake import _FAKE
def get_dataset(config):

    if config.DATASET.DATASET == "360CC":
        return _360CC
    elif config.DATASET.DATASET == "OWN":
        return _OWN
    elif config.DATASET.DATASET == "FAKE":
        return _FAKE
    else:
        raise NotImplemented()