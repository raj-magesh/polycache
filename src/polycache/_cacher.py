import functools
import importlib
import inspect
import os
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, ParamSpec, Self, TypeVar

from xdg_base_dirs import xdg_cache_home

from polycache._handlers import get_handler


def _is_installed(package_name: str) -> bool:
    return importlib.util.find_spec(package_name) is not None


if _is_installed("mne"):
    import mne
if _is_installed("nibabel"):
    import nibabel as nib
if _is_installed("numpy"):
    import numpy as np
if _is_installed("xarray"):
    import xarray as xr
if _is_installed("PIL"):
    from PIL import Image


P = ParamSpec("P")
R = TypeVar("R")

POLYCACHE_HOME = Path(
    os.getenv(
        "POLYCACHE_HOME",
        str(xdg_cache_home() / "polycache"),
    ),
)
DEFAULT_MODE = os.getenv("POLYCACHE_MODE", "normal")


class Cacher:
    def __init__(
        self: Self,
        identifier: str | None = None,
        *,
        path: Path = POLYCACHE_HOME,
        helper: Callable[[Mapping[str, Any]], dict[str, str]] | None = None,
        filetype: str = "auto",
        mode: str = DEFAULT_MODE,
        kwargs_save: Mapping[str, Any] = {},
        kwargs_load: Mapping[str, Any] = {},
    ) -> None:
        """Cache outputs of functions to disk.

        Avoids re-evaluation of (potentially expensive) function when called again.

        When the cacher is called on a function, it computes the output of the function and stores it on disk at the path ``path / identifier``.
        If the function is called again, the cached value is retrieved from disk and returned.

        The identifier can be parametrized by the function inputs.
        For example, if the function takes in the integer x as input, setting the identifier to "{x}.pkl" will result in the filename "2.pkl".
        This is accomplished by calling ``identifier.format(*args, **kwargs)``, which requires that the template arguments have direct string representations.
        For additional flexibility, the cacher offers the ``helper`` argument, which must be a function that takes in the arguments to the function as a dictionary and returns a dictionary mapping template variables to actual values, including potential evaluations.

        Basic usage:

        The following example will cache the output of ``add(3, 5)`` to ``~/output/sums/first_arg_3/second_arg_5.pkl`` as a Python pickle file.

        ```
        from pathlib import Path

        @cache(path=Path.home() / "output", identifier="sums/first_arg_{x}/second_arg_{y}.pkl", filetype="pickle")
        def add(x: int, y: int) -> int:
            return x + y
        ```

        Advanced usage:

        The following example will cache the output of ``add({"three": 3, "five": 5})`` to ``$POLYCACHE_HOME/analysis/keys=three.five/values=3_5/True.pkl``.

        ```
        analysis = "fancy_sum"

        @cache(
            identifier=f"{analysis}/keys={{dict_keys}}/values={{dict_values}}/{{flag}}.pkl",
            helper=lambda kwargs: {
                "dict_keys": ".".join(list(kwargs["x"].keys())),
                "dict_values": "_".join(list(kwargs["x"].values())),
                "flag": kwargs["flag"],
            }
        )
        def add(x: dict[str, float], flag: bool = True) -> float:
            return sum(list(x.values()))
        ```

        Todo:
        ----
            * Support filetype = "auto", which auto-detects filetype based on output class
            * Track progress of :PEP: `501` (https://peps.python.org/pep-0501/) which introduces lazy f-strings. This would allow for a simpler implementation without the ``helper`` argument.

        Args:
        ----
            path: Cache directory to save to/load from. Defaults to the value of the environment variable POLYCACHE_HOME. If the environment variable is not set, defaults to ``~/.cache/polycache``.
            mode: Controls the behavior of the cacher:
                * "normal": If the function output has been previously cached, the stored value is retrieved and returned. If the output has not been previously cached, the function body is run and the output is cached.
                * "readonly": If the function output has been previously cached, the stored value is retrieved and returned. If the output has not been previously cached, the function body is run but the output is NOT cached.
                * "overwrite": The function body is run and the output is cached, overwriting any existing cached output. Existing cached values are not read from.
                * "delete": The function body is run and the output is returned. Any existing cached values are deleted.
                * "ignore": The function body is run and the output is returned. Any existing cached values are ignored.
                Defaults to the value of the environment variable POLYCACHE_MODE. If the environment variable is not set, defaults to "normal".
            identifier: _description_. Defaults to None.
            filetype: Serialization protocol used to cache the files to disk. Supported filetypes include:
                * "auto" (default): Uses one of the following protocols depending on the function output. Requires that the suffix of the 'identifier' is '.npy' for numpy, '.nc' for netCDF4, and '.pkl' for pickle.
                * "numpy": If the function output is a single numpy array, the `numpy.save` function is used.
                * "netCDF4": If the function output is an xarray.DataArray or an xarray.Dataset, the `.to_netcdf` method is used to save the variables to disk.
                * "pickle": All other function outputs are pickled.
                 Defaults to "auto".
            kwargs_save: Keyword arguments passed on to the `.save` method of the `Handler` corresponding to `filetype`. Defaults to {}.
            kwargs_load: Keyword arguments passed on to the `.load` method of the `Handler` corresponding to `filetype`. Defaults to {}.

        """
        self.path = path
        self.path.mkdir(parents=True, exist_ok=True)

        self.mode = mode
        modes = {"normal", "readonly", "overwrite", "delete", "ignore"}
        if mode not in modes:
            error = f"mode {mode} not supported (allowed modes: {modes})"
            raise ValueError(error)

        self.identifier = identifier
        self.helper = helper
        self.filetype = filetype
        self.kwargs_save = kwargs_save
        self.kwargs_load = kwargs_load

    def __call__(
        self: Self,
        func: Callable[P, R],
    ) -> Callable[P, R]:
        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            args_to_format = self._get_args(func, *args, **kwargs)

            if self.helper is not None:
                args_to_format = self.helper(args_to_format)
            identifier = self.identifier.format(**args_to_format)
            filepath = self.path / identifier

            match self.mode:
                case "normal":
                    if filepath.exists():
                        result = self._load(identifier)
                    else:
                        result = func(*args, **kwargs)
                        self._save(result, identifier=identifier)
                case "readonly":
                    if filepath.exists():
                        result = self._load(identifier)
                    else:
                        result = func(*args, **kwargs)
                case "overwrite":
                    result = func(*args, **kwargs)
                    self._save(result, identifier=identifier)
                case "delete":
                    if filepath.exists():
                        filepath.unlink()
                    result = func(*args, **kwargs)
                case "ignore":
                    result = func(*args, **kwargs)
                case _:
                    error = (
                        "mode must be one of 'normal', 'readonly', 'overwrite',"
                        " 'delete', or 'ignore'"
                    )
                    raise ValueError(error)
            return result

        return wrapper

    def _save(self: Self, result: Any, *, identifier: str) -> None:  # type: ignore  # result can be Any
        filepath = self.path / identifier
        filepath.parent.mkdir(parents=True, exist_ok=True)

        if self.filetype == "auto":
            filetype = None

            if _is_installed("numpy") and isinstance(result, np.ndarray):
                filetype = "numpy"
                suffix = ".npy"

            if _is_installed("xarray") and isinstance(
                result, xr.DataArray | xr.Dataset
            ):
                filetype = "netCDF4"
                suffix = ".nc"

            if _is_installed("nibabel") and isinstance(result, nib.nifti1.Nifti1Image):
                filetype = "NIfTI"
                suffix = ".nii.gz"

            if _is_installed("PIL") and isinstance(result, Image.Image):
                filetype = "PIL"
                suffix = None

            if _is_installed("mne"):
                if isinstance(result, mne.io.Raw):
                    filetype = "mne.io.Raw"
                    suffix = ".fif"
                elif isinstance(result, mne.Epochs):
                    filetype = "mne.Epochs"
                    suffix = "-epo.fif"

            if filetype is None:
                filetype = "pickle"
                suffix = ".pkl"

            if suffix is not None and (not filepath.name.endswith(suffix)):
                error = f"identifier must have suffix '{suffix}' if filetype is 'auto'"
                raise ValueError(error)
        else:
            filetype = self.filetype

        handler = get_handler(filetype=filetype)
        handler.save(result=result, path=filepath, **self.kwargs_save)

    def _load(self: Self, identifier: str) -> Any:
        path = self.path / identifier

        if self.filetype == "auto":
            match path.suffix:
                case ".npy":
                    filetype = "numpy"
                case ".nc":
                    filetype = "netCDF4"
                case ".pkl":
                    filetype = "pickle"
                case ".nii" | ".nii.gz":
                    filetype = "NIfTI"
                case ".png" | ".jpg":
                    filetype = "PIL"
                case ".fif":
                    filetype = (
                        "mne.Epochs" if path.stem[-4:] == "-epo" else "mne.io.Raw"
                    )
                case _:
                    raise ValueError
        else:
            filetype = self.filetype

        handler = get_handler(filetype=filetype)
        return handler.load(path=path, **self.kwargs_load)

    def _delete(self: Self, identifier: str) -> None:
        (self.path / identifier).unlink()

    def _get_args(
        self: Self,
        function: Callable[P, R],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> dict[str, Any]:
        signature = inspect.signature(function)
        bound_arguments = signature.bind(*args, **kwargs)
        bound_arguments.apply_defaults()
        return bound_arguments.arguments
