# physt
P(i/y)thon h(i/y)stograms. 

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

## Features

### Implemented

* Input: any numpy-array-like object

### Planned

* Normalization
* Add new values (with heights)
* Underflow / overflow
* Stacked histograms
* Input: pandas.Series, pandas.DataFrame, ...
