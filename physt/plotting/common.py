import numpy as np


def get_data(h, density=False, cumulative=False, flatten=False):
    if density:
        if cumulative:
            data = (h / h.total).cumulative_frequencies
        else:
            data = h.densities
    else:
        if cumulative:
            data = h.cumulative_frequencies
        else:
            data = h.frequencies

    if flatten:
        data = data.flatten()
    return data


def transform_data(data, kwargs):
    transform = kwargs.pop("transform", lambda x:x)
    if not isinstance(transform, np.ufunc):
        transform = np.vectorize(transform)
    return transform(data)