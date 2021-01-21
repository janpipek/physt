from typing import Optional, Container, Tuple, Dict, Any, TYPE_CHECKING, cast

import numpy as np

from physt.histogram1d import Histogram1D, ObjectWithBinning
from physt.binnings import BinningBase, BinningLike, as_binning
from physt.typing_aliases import ArrayLike

if TYPE_CHECKING:
    import physt


class HistogramCollection(Container[Histogram1D], ObjectWithBinning):
    """Experimental collection of histograms.

    It contains (potentially name-addressable) 1-D histograms
    with a shared binning.
    """

    def __init__(
        self,
        *histograms: Histogram1D,
        binning: Optional[BinningLike] = None,
        title: Optional[str] = None,
        name: Optional[str] = None
    ):
        self.histograms = list(histograms)
        if histograms:
            if binning:
                raise ValueError(
                    "When creating collection from histograms, binning is deduced from them."
                )
            self._binning = histograms[0].binning
            if not all(h.binning == self._binning for h in histograms):
                raise ValueError("All histograms should share the same binning.")
        else:
            if binning is None:
                raise ValueError("Either binning or at least one histogram must be provided.")
            self._binning = as_binning(binning)
        self.name = name
        self.title = title or self.name

    def __contains__(self, item):
        try:
            _ = self[item]
            return True
        except KeyError:
            return False

    def __iter__(self):
        return iter(self.histograms)

    def __len__(self):
        return len(self.histograms)

    def copy(self) -> "HistogramCollection":
        # TODO: The binnings are probably not consistent in the copies
        binning_copy = self.binning.copy()
        histograms = [h.copy() for h in self.histograms]
        for histogram in histograms:
            histogram._binning = binning_copy
        return HistogramCollection(*histograms, title=self.title, name=self.name)

    @property
    def binning(self) -> BinningBase:
        return self._binning

    @property
    def axis_name(self) -> str:
        return self.histograms[0].axis_name if self.histograms else "axis0"

    @property
    def axis_names(self) -> Tuple[str]:
        return (self.axis_name,)

    def add(self, histogram: Histogram1D) -> None:
        """Add a histogram to the collection."""
        if self.binning and not self.binning == histogram.binning:
            raise ValueError("Cannot add histogram with different binning.")
        self.histograms.append(histogram)

    def create(
        self, name: str, values, *, weights=None, dropna: bool = True, **kwargs
    ) -> Histogram1D:
        # TODO: Rename!
        init_kwargs: Dict[str, Any] = {"axis_name": self.axis_name}
        init_kwargs.update(kwargs)
        histogram = Histogram1D(binning=self.binning, name=name, **init_kwargs)
        histogram.fill_n(values, weights=weights, dropna=dropna)
        self.histograms.append(histogram)
        return histogram

    def __getitem__(self, item) -> Histogram1D:
        if isinstance(item, str):
            candidates = [h for h in self.histograms if h.name == item]
            if len(candidates) == 0:
                raise KeyError("Collection does not contain histogram named {0}".format(item))
            return candidates[0]
        else:
            return self.histograms[item]

    def __eq__(self, other) -> bool:
        return (
            (type(other) == HistogramCollection)
            and (len(other) == len(self))
            and all((h1 == h2) for h1, h2 in zip(self.histograms, other.histograms))
        )

    def normalize_bins(self, inplace: bool = False) -> "HistogramCollection":
        """Normalize each bin in the collection so that the sum is 1.0 for each bin.

        Note: If a bin is zero in all collections, the result will be inf.
        """
        col = self if inplace else self.copy()
        sums = self.sum().frequencies
        for h in col.histograms:
            h.set_dtype(float)
            h._frequencies /= sums
            h._errors2 /= sums ** 2  # TODO: Does this make sense?
        return col

    def normalize_all(self, inplace: bool = False) -> "HistogramCollection":
        """Normalize all histograms so that total content of each of them is equal to 1.0."""
        col = self if inplace else self.copy()
        for h in col.histograms:
            h.normalize(inplace=True)
        return col

    def sum(self) -> Histogram1D:
        """Return the sum of all contained histograms."""
        if not self.histograms:
            return Histogram1D(
                data=np.zeros((self.binning.bin_count)), dtype=np.int64, binning=self.binning
            )
        return cast(Histogram1D, sum(self.histograms))

    @property
    def plot(self) -> "physt.plotting.PlottingProxy":
        """Proxy to plotting.

        This attribute is a special proxy to plotting. In the most
        simple cases, it can be used as a method. For more sophisticated
        use, see the documentation for physt.plotting package.
        """
        from physt.plotting import PlottingProxy

        return PlottingProxy(self)

    @classmethod
    def multi_h1(cls, a_dict: Dict[str, ArrayLike], bins=None, **kwargs) -> "HistogramCollection":
        """Create a collection from multiple datasets."""
        from physt.binnings import calculate_bins

        mega_values = np.concatenate(list(a_dict.values()))
        binning = calculate_bins(mega_values, bins, **kwargs)

        title = kwargs.pop("title", None)
        name = kwargs.pop("name", None)

        collection = HistogramCollection(binning=binning, title=title, name=name)
        for key, value in a_dict.items():
            collection.create(key, value)
        return collection

    @classmethod
    def from_dict(cls, a_dict: Dict[str, Any]) -> "HistogramCollection":
        from physt.io import create_from_dict

        histograms = (
            cast(Histogram1D, create_from_dict(item, "HistogramCollection", check_version=False))
            for item in a_dict["histograms"]
        )
        return HistogramCollection(*histograms)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "histogram_type": "histogram_collection",
            "histograms": [h.to_dict() for h in self.histograms],
        }

    def to_json(self, path: Optional[str] = None, **kwargs) -> str:
        """Convert to JSON representation.

        Parameters
        ----------
        path: Where to write the JSON.

        Returns
        -------
        The JSON representation.
        """
        from .io import save_json

        return save_json(self, path, **kwargs)
