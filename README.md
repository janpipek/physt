# physt

P(i/y)thon h(i/y)stograms. Inspired (and based on) numpy.histogram, but designed for humans(TM) on steroids(TM).

The goal is to unify different concepts of histograms as occurring in numpy, pandas, matplotlib, ROOT, etc.
and to create one representation that is easily manipulated with from the data point of view and at the same time provides
nice integration into IPython notebook and various plotting options. In short, whatever you want to do with histograms,
**physt** aims to be on your side.

[![Join the chat at https://gitter.im/physt/Lobby](https://badges.gitter.im/physt/physt.svg)](https://gitter.im/physt/physt) [![PyPI version](https://badge.fury.io/py/physt.svg)](https://badge.fury.io/py/physt)
[![Anaconda-Server Badge](https://anaconda.org/janpipek/physt/badges/version.svg)](https://anaconda.org/janpipek/physt)
[![Anaconda-Server Badge](https://anaconda.org/janpipek/physt/badges/license.svg)](https://anaconda.org/janpipek/physt)

## Simple example

```python
from physt import histogram

# Create the sample
heights = [160, 155, 156, 198, 177, 168, 191, 183, 184, 179, 178, 172, 173, 175,
           172, 177, 176, 175, 174, 173, 174, 175, 177, 169, 168, 164, 175, 188,
           178, 174, 173, 181, 185, 166, 162, 163, 171, 165, 180, 189, 166, 163,
           172, 173, 174, 183, 184, 161, 162, 168, 169, 174, 176, 170, 169, 165]

hist = histogram(heights, 10)    # <--- get the histogram data
hist.plot()                      # <--- and plot it
```

![Heights plot](doc/heights.png)

## 2D example

```python
from physt import h2
import seaborn as sns

iris = sns.load_dataset('iris')
iris_hist = h2(iris["sepal_length"], iris["sepal_width"], "human", (12, 7), name="Iris")
iris_hist.plot(show_zero=False, cmap=cm.gray_r, show_values=True);
```

![Iris 2D plot](doc/iris-2d.png)

## 3D directional example

```python
import numpy as np
from physt import special

# Generate some sample data 
data = np.empty((n, 3))
data[:,0] = np.random.normal(0, 1, n)
data[:,1] = np.random.normal(0, 1.3, n)
data[:,2] = np.random.normal(1, .6, n)

# Get histogram data (in spherical coordinates)
h = special.spherical_histogram(data)                 

# And plot its projection on a globe
h.projection("theta", "phi").plot.globe_map(density=True, figsize=(7, 7), cmap="rainbow")   
```

![Directional 3D plot](doc/globe.png)

See more in docstring's and notebooks:

- Basic tutorial: <http://nbviewer.jupyter.org/github/janpipek/physt/blob/master/doc/Tutorial.ipynb>
- Binning: <http://nbviewer.jupyter.org/github/janpipek/physt/blob/master/doc/Binning.ipynb>
- Bokeh plots: <http://nbviewer.jupyter.org/github/janpipek/physt/blob/master/doc/Bokeh%20examples.ipynb>
- 2D histograms: <http://nbviewer.jupyter.org/github/janpipek/physt/blob/master/doc/2D%20Histograms.ipynb>
- Special histograms (polar, spherical, cylindrical - *beta*): <http://nbviewer.jupyter.org/github/janpipek/physt/blob/master/doc/Special%20histograms.ipynb>
- Adaptive histograms: <http://nbviewer.jupyter.org/github/janpipek/physt/blob/master/doc/Adaptive%20histogram.ipynb>
- Use dask for large (not "big") data: <http://nbviewer.jupyter.org/github/janpipek/physt/blob/master/doc/Dask.ipynb>

## Installation

Using pip:

`pip install physt`

Using conda (not always up-to-date):

`conda install -c janpipek physt`

## Features

### Implemented

* 1D histograms
* 2D histograms
* ND histograms
* Some special histograms
  - 2D polar coordinates (with plotting)
  - 3D spherical / cylindrical coordinates (beta)
* Adaptive rebinning for on-line filling of unknown data (beta)
* Non-consecutive bins
* Memory-effective histogramming of dask arrays (beta)
* Understands any numpy-array-like object
* Keep underflow / overflow / missed bins
* Basic numeric operations (* / + -)
* Items / slice selection (including mask arrays)
* Add new values (fill, fill_n)
* Cumulative values, densities
* Simple statistics for original data (mean, std, sem)
* Simple plotting (matplotlib, bokeh)
* Algorithms for optimized binning
  - human-friendly
  - mathematical
* IO, conversions
  - I/O xarray.DataSet
  - I/O JSON
  - O pandas.DataFrame

### Planned
* Rebinning
  - using reference to original data?
  - merging bins
* Statistics (based on original data)?
* Stacked histograms (with names)
* More plotting backends

### Not planned
* Kernel density estimates - use your favourite statistics package (like `seaborn`)
* Rebinning using interpolation - it should be trivial to use `rebin` (<https://github.com/jhykes/rebin>) with physt

Rationale (for both): physt is dumb, but precise.

## Dependencies

- Python 3.5 targeted, 2.7 passes unit tests
- numpy
- (optional) matplotlib - simple output
- (optional) bokeh - simple output
- (optional) xarray - I/O
- (optional) astropy - additional binning algorithms
- (testing) py.test, pandas
- (docs) sphinx, sphinx_rtd_theme, ipython

## Contribution

I am looking for anyone interested in using / developing physt. You can contribute by reporting errors, implementing missing features and suggest new one.
