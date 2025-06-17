# cfbooklet

<p align="center">
    <em>CF conventions multi-dimensional array storage on top of Booklet</em>
</p>

[![build](https://github.com/mullenkamp/cfbooklet/workflows/Build/badge.svg)](https://github.com/mullenkamp/cfbooklet/actions)
[![codecov](https://codecov.io/gh/mullenkamp/cfbooklet/branch/master/graph/badge.svg)](https://codecov.io/gh/mullenkamp/cfbooklet)
[![PyPI version](https://badge.fury.io/py/cfbooklet.svg)](https://badge.fury.io/py/cfbooklet)

---

**Documentation**: <a href="https://mullenkamp.github.io/cfbooklet/" target="_blank">https://mullenkamp.github.io/cfbooklet/</a>

**Source Code**: <a href="https://github.com/mullenkamp/cfbooklet" target="_blank">https://github.com/mullenkamp/cfbooklet</a>

---

## Development

### Coordinate variables
Must be 1D. 
They should have an "ordered" parameter (bool) that defined whether the coord should always be ordered. Int, float, and datetime should default to True. Only string and category dtypes should default to False.
There should be a "regular" parameter (bool) with an associated "step" parameter (int or float). It should work similarly to np.arange. Only ints, floats, and datetimes can use this. 
~~Should I add a "unique" parameter (bool)? Maybe I should just enforce this normally?~~ It should enforce uniqueness in the coords.
There can be a groupby method datasets that would use the rechunker. The rechunker would have the groupby dims set to 1 and the other dims set to the full length.

#### Multi-dimensional coords
It is possible to create a composite index from multiple 1D coords. But it seems best to implement this type of thing on top of sqlite (or something equivalent). 
Keeping each coord 1D makes implementations quite a bit simpler. 

## License

This project is licensed under the terms of the Apache Software License 2.0.
