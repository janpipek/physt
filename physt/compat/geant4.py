from ..histogram1d import Histogram1D
from ..binnings import fixed_width_binning
import codecs
import numpy as np


def load_h1_csv(path):
	meta = {}
	data = []
	with codecs.open(path, encoding="ASCII") as in_file:
		for line in in_file:
			if line.startswith("#"):
				key, value = line[1:].strip().split(" ", 1)
				meta[key] = value
			else:
				try:
					data.append([float(frag) for frag in line.split(",")])
				except:
					pass
	data = np.asarray(data)
	_, bin_count, min_, max_ = meta["axis"].split()
	bin_count = int(bin_count)
	min_ = float(min_)
	max_ = float(max_)
	binning = fixed_width_binning(None, bin_width=(max_ - min_) / bin_count, range=(min_, max_))
	h = Histogram1D(binning, name=meta["title"])
	h._frequencies = data[1:-1,1]
	h._errors2 = data[1:-1,2]
	h.underflow = data[0,1]
	h.overflow = data[-1,1]
	h._stats = {
		"sum" : data[1:-1,3].sum(),
		"sum2" : data[1:-1,4].sum()
	}
	return h
