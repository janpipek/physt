from .histogram_nd import HistogramND
from . import binning, histogram_nd
import numpy as np


class PolarHistogram(HistogramND):
    def __init__(self, bins, frequencies=None, **kwargs):
        if not "axis_names" in kwargs:
            kwargs["axis_names"] = ("r", "phi")
        if "dim" in kwargs:
            kwargs.pop("dim")
        super(PolarHistogram, self).__init__(2, bins=bins, frequencies=frequencies, **kwargs)

    @property
    def bin_sizes(self):
        sizes = self.get_bin_right_edges(0) ** 2 - self.get_bin_left_edges(0) ** 2
        sizes = np.outer(sizes, self.get_bin_widths(1))
        return sizes

    def find_bin(self, value, axis=None):
        raise NotImplementedError()

    def plot(self, histtype="map", density=False, backend="matplotlib", **kwargs):
        color = kwargs.pop("color", "frequency")
        show_zero = kwargs.pop("show_zero", True)
        cmap = kwargs.pop("cmap", "Greys")

        if backend == "matplotlib":
            import matplotlib.pyplot as plt
            import matplotlib.cm as cm
            import matplotlib.colors as colors

            if density:
                dz = self.densities.flatten()
            else:
                dz = self.frequencies.flatten()

            if color == "frequency":
                cmap_max = kwargs.pop("cmap_max", dz.max())
                cmap_min = kwargs.pop("cmap_min", 0)
                if cmap_min == "min":
                    cmap_min = dz.min()
                norm = colors.Normalize(cmap_min, cmap_max, clip=True)

                if isinstance(cmap, str):
                    cmap = plt.get_cmap(cmap)
                colors = cmap(norm(dz))
            else:
                colors = color   # TODO: does not work for map

            if histtype == "map":
                figsize = kwargs.pop("figsize", None)
                fig = plt.figure(figsize=figsize)
                ax = fig.add_subplot(111, projection='polar')

                # ypos, xpos = (arr.flatten() for arr in self.get_bin_left_edges())
                # # if show_values:
                # #     text_x, text_y = (arr.flatten() for arr in self.get_bin_centers())
                #
                # dy, dx = (arr.flatten() for arr in self.get_bin_widths())
                #
                # for i in range(len(xpos)):
                #     bin_color = colors[i]
                #
                #     if dz[i] > 0 or show_zero:
                #         rect = plt.Rectangle([xpos[i], ypos[i]], dx[i], dy[i],
                #                              facecolor=bin_color, edgecolor=kwargs.get("grid_color", cmap(0.5)),
                #                              lw=kwargs.get("lw", 0.5))
                #         ax.add_patch(rect)
                # ax.autoscale()
                rpos, phipos = (arr.flatten() for arr in self.get_bin_left_edges())
                # _, phipos = (arr.flatten() for arr in self.get_bin_centers())
                dr, dphi  = (arr.flatten() for arr in self.get_bin_widths())
                rmax, _ =  (arr.flatten() for arr in self.get_bin_right_edges())

                for i in range(len(rpos)):
                    if dz[i] > 0 or show_zero:
                        bin_color = colors[i]
                        bars = ax.bar(phipos[i], dr[i], width=dphi[i], bottom=rpos[i], color=bin_color, edgecolor=kwargs.get("grid_color", cmap(0.5)), lw=kwargs.get("lw", 0.5))
                    #
                    #     if dz[i] > 0 or show_zero:
                    #         rect = plt.Rectangle([xpos[i], ypos[i]], dx[i], dy[i],
                    #                              facecolor=bin_color, edgecolor=kwargs.get("grid_color", cmap(0.5)),
                    #                              lw=kwargs.get("lw", 0.5))
                    #         ax.add_patch(rect)
                ax.set_rmax(rmax.max())
            else:
                raise RuntimeError("Unsupported hist type")
            return ax
        else:
            raise RuntimeError("Unsupported hist type")






def polar_histogram(xdata, ydata, radial_bins="human", phi_bins=16, *args, **kwargs):
    rdata = np.hypot(xdata, ydata)
    phidata = np.arctan2(xdata, ydata)
    data = np.concatenate([rdata[:, np.newaxis], phidata[:, np.newaxis]], axis=1)
    dropna = kwargs.pop("dropna", False)
    if isinstance(phi_bins, int):
        phi_range = (-np.pi, np.pi)
        if "range" in "kwargs":
            phi_range = kwargs["range"][1]
        phi_bins = np.linspace(*phi_range, phi_bins + 1)

    if dropna:
        data = data[~np.isnan(data).any(axis=1)]
    bins = binning.calculate_bins_nd(data, [radial_bins, phi_bins], *args, check_nan=not dropna, **kwargs)

    # Prepare remaining data
    weights = kwargs.pop("weights", None)
    frequencies, errors2, missed = histogram_nd.calculate_frequencies(data, ndim=2,
                                                                  bins=bins,
                                                                  weights=weights)
    return PolarHistogram(bins=bins, frequencies=frequencies, errors2=errors2, missed=missed)

