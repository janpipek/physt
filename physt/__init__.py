from . import binning, bin_utils, histogram1d

import numpy as np

__version__ = str('0.1.0')


def histogram(data=None, _=None, *args, **kwargs):
    """Facade function to create histograms.

    This proceeds in three steps:
    1) Based on magical parameter _, construct bins for the histogram
    2) Calculate frequencies for the bins
    3) Construct the histogram object itself

    Parameters
    ----------
    data : array_like, optional
        Container of all the values (tuple, list, np.ndarray, pd.Series)
    _: int or sequence of scalars or callable or str, optional
        If iterable => the bins themselves
        If int => number of bins for default binning
        If callable => use binning method (+ args, kwargs)
        If string => use named binning method (+ args, kwargs)
    weights: array_like, optional
        (as numpy.histogram)
    keep_missed: bool, optional
        store statistics about how many values were lower than limits and how many higher than limits (default: True)

    Other numpy.histogram parameters are excluded, see the methods of the Histogram1D class itself.

    Returns
    -------
    Histogram1D or callable
        - if data is None -> callable

    See Also
    --------
    numpy.histogram
    """
    if isinstance(data, np.ndarray):
        weights = kwargs.get("weights", None)
        keep_missed = kwargs.get("keep_missed", True)

        # Get binning
        if _ is None:
            bin_count = kwargs.pop("bins", binning.ideal_bin_count(data=data))
            bins = binning.numpy_like(data, bin_count, *args, **kwargs)
        elif isinstance(_, int):
            bins = binning.numpy_like(data, _, *args, **kwargs)
        elif isinstance(_, str):
            method = binning.binning_methods[_]
            bins = method(data, *args, **kwargs)
        elif callable(_):
            bins = _(data, *args, **kwargs)
        elif np.iterable(_):
            bins = _
        else:
            raise RuntimeError("Binning {0} not understood.".format(_))
        bins = bin_utils.make_bin_array(bins)

        # Get frequencies
        frequencies, errors2, underflow, overflow = histogram1d.calculate_frequencies(data,
                                                                                      bins=bins,
                                                                                      weights=weights)

        # Construct the object
        keep_missed = kwargs.get("keep_missed", True)
        if not keep_missed:
            underflow = 0
            overflow = 0
        return histogram1d.Histogram1D(bins=bins, frequencies=frequencies, errors2=errors2, overflow=overflow,
                                       underflow=underflow, keep_missed=keep_missed)
    else:
        return histogram(np.array(data), _, **kwargs)