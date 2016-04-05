from . import binning, bin_utils, histogram1d

import numpy as np

__version__ = str('0.2.1')


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
    keep_missed: Optional[bool]
        store statistics about how many values were lower than limits and how many higher than limits (default: True)
    name: str
        name of the histogram
    axis_name: str
        name of the variable on x axis

    Other numpy.histogram parameters are excluded, see the methods of the Histogram1D class itself.

    Returns
    -------
    Histogram1D

    See Also
    --------
    numpy.histogram

    Examples
    --------
    """
    if isinstance(data, tuple) and isinstance(data[0], str):    # Works for groupby DataSeries
        return histogram(data[1], _, *args, name=data[0], **kwargs)
    elif type(data).__name__ == "DataFrame":
        raise RuntimeError("Cannot create histogram from a pandas DataFrame. Use Series.")
    else:
        # Collect arguments (not to send them to binning algorithms)
        weights = kwargs.pop("weights", None)
        keep_missed = kwargs.pop("keep_missed", True)
        name = kwargs.pop("name", None)
        axis_name = kwargs.pop("axis_name", None)

        # Convert to array
        array = np.asarray(data).flatten()

        # Get binning
        bins = binning.calculate_bins(array, _, *args, **kwargs)

        # Get frequencies
        frequencies, errors2, underflow, overflow = histogram1d.calculate_frequencies(array,
                                                                                      bins=bins,
                                                                                      weights=weights)

        # Construct the object
        if not keep_missed:
            underflow = 0
            overflow = 0
        if hasattr(data, "name") and not axis_name:
            axis_name = data.name
        return histogram1d.Histogram1D(bins=bins, frequencies=frequencies, errors2=errors2, overflow=overflow,
                                       underflow=underflow, keep_missed=keep_missed, name=name, axis_name=axis_name)