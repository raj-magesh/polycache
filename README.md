# Polycache

This Python package provides a simple decorator that allows you to cache function outputs to disk.

1. Evaluate an expensive function once.
2. Store its output on disk.
3. Whenever the function is called again, retrieve the cached output from disk instead of evaluating it again.

## Features

- **Simple to use**: Apply a single decorator to your function to enable powerful caching
- **Native file formats**: The outputs of functions are serialized to disk using native file formats.
    - For example, [`numpy` arrays](https://numpy.org/doc/stable/reference/arrays.ndarray.html#arrays-ndarray) are saved as [`.npy` files](https://numpy.org/doc/stable/reference/generated/numpy.save.html).
    - As a fallback, function outputs can be saved as generic [`pickle` files](https://docs.python.org/3/library/pickle.html) (`.pkl`).
- **Parameterizable**: The filepaths where function outputs are saved can be parameterized by the function arguments.
    - For example, when `add(x, y)` is called with `x=2` and `y=3`, its output can be automatically saved to `x=2_y=3_output.pkl`.
    - Nested directories are also possible: e.g. the same output could be saved to `x=2/y=3/output.pkl`.
- **Customizable**: Keyword arguments can be passed to control the serialization/deserialization processes.
    - For example, if the function returns multiple numpy arrays (handled by [`numpy.savez`](https://numpy.org/doc/stable/reference/generated/numpy.savez.html#numpy.savez) internally), you could pass `compress=True` to enable compression.
- **Extensible**: You can define custom `save` and `load` functions to use your own file formats.
- **Lazy**: When supported by the file format, cached results are lazy-loaded by default to speed up the function call and save memory.
    - For example, [`xarray.Dataset`](https://docs.xarray.dev/en/stable/generated/xarray.Dataset.html) outputs are loaded from `.nc` netCDF-4 files using [`xarray.open_dataset`](https://docs.xarray.dev/en/stable/generated/xarray.open_dataset.html) (lazy) instead of [`xarray.load_dataarray`](https://docs.xarray.dev/en/stable/generated/xarray.load_dataset.html) (which eagerly loads the file contents into memory).

## Usage

The following example will cache the output of `add(3, 5)` to `~/output/sums/first_arg_3/second_arg_5.pkl` as a Python pickle file.

```python
from pathlib import Path

@cache(
   path=Path.home() / "output",
   identifier="sums/first_arg_{x}/second_arg_{y}.pkl",
   filetype="pickle",
)
def add(x: int, y: int) -> int:
    return x + y
```

The following example will cache the output of `add({"three": 3, "five": 5})` to `$POLYCACHE_HOME/analysis/keys=three.five/values=3_5/True.pkl`.

```python
analysis = "fancy_sum"

@cache(
    f"{analysis}/keys={{dict_keys}}/values={{dict_values}}/{{flag}}.pkl",
    helper=lambda kwargs: {
        "dict_keys": ".".join(list(kwargs["x"].keys())),
        "dict_values": "_".join(list(kwargs["x"].values())),
        "flag": kwargs["flag"],
    },
)
def add(x: dict[str, float], flag: bool = True) -> float:
    return sum(list(x.values()))
```

## Supported file formats

## Acknowledgments

The initial inspiration for this decorator came from a similar implementation called [result_caching](https://github.com/brain-score/result_caching) used internally by the [Brain-Score](http://www.brain-score.org) project. While this was handy, it was also rather inflexible, so I forked it and added features to support my use cases.
