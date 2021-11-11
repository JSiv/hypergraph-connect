import torch.nn as nn


def process(data):
    data = (data + 1.) / 2.
    # clamps data in to a range of 0, 1
    data.clamp_(0, 1)
    return data


def sample(data_loader):
    while True:
        for data in data_loader:
            yield data


def spectral_norm(module, use_spectral_norm=True):
    if not use_spectral_norm:
        return module
    else:
        return nn.utils.spectral_norm(module)
