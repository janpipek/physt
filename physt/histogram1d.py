import numpy as np


class Histogram1D(object):
    """Representation of one-dimensional histogram.
    """
    def __init__(self, bins, values=None):
        bins = np.array(bins)
        if bins.ndim == 1:       # Numpy-style
            self._bins = np.hstack((bins[:-1,np.newaxis], bins[1:,np.newaxis]))
        elif bins.ndim == 2:     # Tuple-style
            if bins.shape[1] != 2:
                raise RuntimeError("")
            self._bins = bins
        else:
            raise RuntimeError("Unexpected format of bins.")

        if values is None:
            self._values = np.zeros(self._bins.shape[0])
        else:
            values = np.array(values)
            if values.shape != (self._bins.shape[0],):
                raise RuntimeError("Values must have same dimension as bins.")
            self._values = values

    @property
    def bins(self):
        return self._bins

    def __getitem__(self, i):
        """Select subhistogram or get one bin."""
        if isinstance(i, int):
            return self.bins[i], self.values[i]
        # elif isinstance(i, slice)) or isinstance(i, np.ndarray):
        else:
            return self.__class__(self.bins[i], self.values[i])

    @property
    def values(self):
        return self._values

    @property
    def numpy_bins(self):
        return np.concatenate((self.left_edges, self.right_edges[-1:]), axis=0)

    @property
    def nbins(self):
        return self.bins.shape[0]

    @property
    def left_edges(self):
        return self.bins[:,0]

    @property
    def right_edges(self):
        return self.bins[:,1]

    @property
    def centers(self):
        return (self.left_edges + self.right_edges) / 2

    @property
    def widths(self):
        return self.right_edges - self.left_edges

    def plot(self, histtype='bar', backend="matplotlib", axis=None, **kwargs):
        """Plot the histogram.

        :param histtype: ‘bar’ | ‘step’ | 'scatter'
        """
        # TODO: See http://matplotlib.org/1.5.0/examples/api/filled_step.html
        if backend == "matplotlib":
            if not axis:
                import matplotlib.pyplot as plt
                _, axis = plt.subplots()
            if histtype == "step":
                x = np.concatenate(([0.0], self.numpy_bins), axis=0)
                y = np.concatenate(([0.0], self.values, [0]), axis=0)
                axis.step(x, y, where="post", **kwargs)
            elif histtype == "bar":
                # TODO: Fix for non-connected histograms
                axis.bar(self.left_edges, self.values, self.widths, **kwargs)
            elif histtype == "scatter":
                axis.scatter(self.centers, self.values, **kwargs)
            else:
                raise RuntimeError("Unknown histogram type: {0}".format(histtype))
        else:
            raise RuntimeError("Only matplotlib supported at the moment.")

    def to_dataframe(self):
        """Convert to pandas DataFrame."""
        import pandas as pd
        df = pd.DataFrame({"left" : self.left_edges, "right" : self.right_edges, "value" : self.values})
        return df

    def __repr__(self):
        return "{0}(bins={1})".format(
            self.__class__.__name__, self.bins.shape[0]) #, self.total, self.underflow, self.overflow)