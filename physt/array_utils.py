import numpy as np


def make_2d_array(ndim:int, *arrays):
    if ndim < 2:
        raise ValueError("make_2d_array")
    if len(arrays) == 1:
        array = np.asarray(arrays[0])
    elif len(arrays) == ndim:    
        numpy_arrays = [np.asarray(a) for a in arrays]
        array = np.asarray(numpy_arrays).T # TODO: More clever concat!
    else:
        raise ValueError("make_2d_array({0}, ...) requires 1 or {0} arguments, {1} supplied.".format(ndim, len(arrays)))

    if array.ndim != 2 or array.shape[1] != ndim:
        raise ValueError("At least one of the arrays supplied was invalid.")
    return array
