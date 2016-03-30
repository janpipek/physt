# physt
P(i/y)thon h(i/y)stograms. Based on numpy.histogram but designed for humans(TM) on steroids(TM).

## Simple example

```python
from physt import histogram

heights = [160, 155, 156, 198, 177, 168, 191, 183, 184, 179, 178, 172, 173, 175,
           172, 177, 176, 175, 174, 173, 174, 175, 177, 169, 168, 164, 175, 188,
           178, 174, 173, 181, 185, 166, 162, 163, 171, 165, 180, 189, 166, 163,
           172, 173, 174, 183, 184, 161, 162, 168, 169, 174, 176, 170, 169, 165]
           
hist = histogram(heights, 10)
hist.plot()
```

![Heights plot](doc/heights.png)

See more in <https://github.com/janpipek/physt/blob/master/doc/Tutorial.ipynb>

## Features

### Implemented

* 1D histograms
* Input: any numpy-array-like object
* Keep underflow / overflow
* Basic numeric operations (* / + -)
* Items / slice selection [including mask arrays]
* Add new values (fill)
* Cumulative values, densities
* Simple plotting (matplotlib)

### Planned

* Algorithms for optimized binning
  - human-friendly
  - mathematical
* Rebinning
  - using reference to original data
  - merging bins
* Statistics (based on original data)?
* Stacked histograms (with names)
* Input: pandas.Series, pandas.DataFrame, ...
* More plotting backends
* 2D histograms, (ND)-histograms
