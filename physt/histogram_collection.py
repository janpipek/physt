from typing import Optional

from .histogram1d import Histogram1D
from .binnings import BinningBase


class HistogramCollection:
    """Experimental collection of histograms."""
    def __init__(self,
                 *histograms: Histogram1D,
                 binning: Optional[BinningBase] = None):
        self.histograms = list(histograms)
        if histograms:
            if binning:
                raise ValueError("")
            self._binning = histograms[0].binning
            if not all(h.binning == self._binning for h in histograms):
                raise ValueError("All histogram should share the same binning.")
        else:
            self._binning = binning

    @property
    def binning(self) -> BinningBase:
        return self._binning

    @property
    def axis_name(self) ->  Optional[str]:
        return self.histograms and self.histograms[0].axis_name or None

    def add(self, histogram: Histogram1D):
        if not self.binning == histogram.binning:
            raise ValueError("Cannot add histogram with different binning.")
        self.histograms.append(histogram)

    def create(self, name: str, values, *, weights=None, dropna: bool = True, **kwargs):
        init_kwargs = {
            "axis_name": self.axis_name
        }
        init_kwargs.update(kwargs)
        histogram = Histogram1D(binning=self.binning, name=name, axis_name=self.axis_name, **init_kwargs)
        histogram.fill_n(values, weights=weights, dropna=dropna)
        self.histograms.append(histogram)
        return histogram

    def __getitem__(self, item) -> Histogram1D:
        if isinstance(item, int):
            return self.histograms[item]
        else:
            candidates = [h for h in self.histograms if h.name == item]
            if len(candidates) == 0:
                raise KeyError("Collection does not contain histogram named {0}".format)
            return candidates[0]
