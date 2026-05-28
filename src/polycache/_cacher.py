import functools
import importlib
import importlib.util
import inspect
import os
import tempfile
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from xdg_base_dirs import xdg_cache_home

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping
    from types import ModuleType

OPTIONAL_PACKAGES = {
    package: importlib.util.find_spec(package) is not None
    for package in ("mne", "edfio", "nibabel", "numpy", "xarray", "PIL")
}
if OPTIONAL_PACKAGES["edfio"]:
    import edfio
if OPTIONAL_PACKAGES["mne"]:
    import mne
if OPTIONAL_PACKAGES["nibabel"]:
    from nibabel.spatialimages import SpatialImage
if OPTIONAL_PACKAGES["numpy"]:
    import numpy as np
if OPTIONAL_PACKAGES["xarray"]:
    import xarray as xr
if OPTIONAL_PACKAGES["PIL"]:
    from PIL.Image import Image

Filetype = Literal[
    "EDF",
    "mne.Epochs",
    "mne.io.Raw",
    "netCDF4",
    "NIfTI",
    "numpy",
    "numpy.z",
    "pickle",
    "PIL",
]

Mode = Literal["normal", "readonly", "overwrite", "ignore", "delete"]

SUFFIXES: dict[Filetype, tuple[str, ...]] = {
    "EDF": (".edf",),
    "mne.Epochs": ("-epo.fif",),
    "mne.io.Raw": (".fif",),
    "netCDF4": (".nc",),
    "NIfTI": (".nii.gz", ".nii"),
    "numpy.z": (".npz",),
    "numpy": (".npy",),
    "pickle": (".pkl",),
    "PIL": (".png", ".jpg"),
}

POLYCACHE_HOME = Path(
    os.getenv(
        "POLYCACHE_HOME",
        str(xdg_cache_home() / "polycache"),
    ),
)


def cache[**P, R](  # noqa: C901, PLR0913
    identifier: str,
    *,
    path: Path = POLYCACHE_HOME,
    helper: Callable[[Mapping[str, Any]], dict[str, str]] | None = None,
    filetype: Filetype | None = None,
    mode: Mode = "normal",
    kwargs_save: Mapping[str, Any] | None = None,
    kwargs_load: Mapping[str, Any] | None = None,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Cache outputs of functions to disk.

    Avoids re-evaluation of (potentially expensive) function when called again.

    When the cacher is called on a function for the first time, it computes the
    output of the function and stores it on disk at the path ``path /
    identifier``. Whenever the function is called again, the cached value is
    retrieved from disk and returned.

    The identifier can be parameterized by the function inputs. For example, if
    the function takes in the integer x as input, setting the identifier to
    "{x}.pkl" will result in the filename "2.pkl". This is accomplished by
    calling ``identifier.format(*args, **kwargs)``, which requires that the
    template arguments have direct string representations. For additional
    flexibility, the cacher offers the ``helper`` argument, which must be a
    function that takes in the arguments to the function as a dictionary and
    returns a dictionary mapping template variables to actual values, including
    potential evaluations.

    Parameters
    ----------
    identifier
        Subpath of the file to be cached. As shown in the examples below,
        this string can be flexibly parameterized by the function arguments
        at runtime.
    helper
        Allows parameterization of the identifier by the function arguments.
        See the examples below for usage.
    path
        Cache directory to save to/load from. If not specified, defaults to
        ``$POLYCACHE_HOME``, ``$XDG_CACHE_HOME/polycache``, and
        ``$HOME/.cache/polycache``, in that order.
    mode
        Controls the behavior of the cacher:
            * "normal" (default): If the function output has been previously cached, the
                stored value is retrieved and returned. If the output has not
                been previously cached, the function body is run and the output
                is cached.
            * "readonly": If the function output has been previously cached,
                the stored value is retrieved and returned. If the output has
                not been previously cached, the function body is run but the
                output is NOT cached.
            * "overwrite": The function body is run and the output is
                cached, overwriting any existing cached output. Existing cached
                values are not read.
            * "delete": The function body is run and the output is returned.
                Any existing cached values are deleted.
            * "ignore": The function body is run and the output is returned.
                Any existing cached values are ignored.
    filetype
        Protocols used to save/load the files to disk. Supported
        filetypes include:
            * None (default): Uses one of the following filetypes depending
                on the output of the function.
            * "numpy": If the function output is a single numpy array, the
                `numpy.save` function is used.
            * "netCDF4": If the function output is an xarray.DataArray or an
                xarray.Dataset, the `.to_netcdf` method is used to save the
                variables to disk.
            * "pickle": All other function outputs are pickled.
                Defaults to "auto".
    kwargs_save
        Optional keyword arguments passed to the serialization function.
    kwargs_load
        Optional keyword arguments passed to the deserialization function.

    Returns
    -------
        Decorated function.

    Examples
    --------
    The following example will cache the output of ``add(3, 5)`` to
    ``~/output/sums/first_arg_3/second_arg_5.pkl`` as a Python pickle file.

    >> from pathlib import Path
    >>
    >> @cache(
    >>    path=Path.home() / "output",
    >>    identifier="sums/first_arg_{x}/second_arg_{y}.pkl",
    >>    filetype="pickle",
    >> )
    >> def add(x: int, y: int) -> int:
    >>     return x + y

    The following example will cache the output of ``add({"three": 3, "five":
    5})`` to ``$POLYCACHE_HOME/analysis/keys=three.five/values=3_5/True.pkl``.

    >> analysis = "fancy_sum"
    >>
    >> @cache(
    >>     f"{analysis}/keys={{dict_keys}}/values={{dict_values}}/{{flag}}.pkl",
    >>     helper=lambda kwargs: {
    >>         "dict_keys": ".".join(list(kwargs["x"].keys())),
    >>         "dict_values": "_".join(list(kwargs["x"].values())),
    >>         "flag": kwargs["flag"],
    >>     },
    >> )
    >> def add(x: dict[str, float], flag: bool = True) -> float:
    >>     return sum(list(x.values()))

    Todo
    ----
        * Track progress of :PEP: `501` (https://peps.python.org/pep-0501/)
            which introduces lazy f-strings. This would allow for a simpler
            implementation without the ``helper`` argument.

    """

    def decorator[**P, R](func: Callable[P, R]) -> Callable[P, R]:  # noqa: C901
        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            signature = inspect.signature(func)
            bound_arguments = signature.bind(*args, **kwargs)
            bound_arguments.apply_defaults()
            args_to_format = bound_arguments.arguments

            if helper is not None:
                args_to_format = helper(args_to_format)

            filepath = path / identifier.format(**args_to_format)

            match mode:
                case "normal":
                    if filepath.exists():
                        result = _load(
                            filepath,
                            filetype=filetype,
                            **kwargs_load if kwargs_load is not None else {},
                        )
                    else:
                        result = func(*args, **kwargs)
                        _save(
                            result,
                            filepath=filepath,
                            filetype=filetype,
                            **kwargs_save if kwargs_save is not None else {},
                        )
                case "readonly":
                    if filepath.exists():
                        result = _load(
                            filepath,
                            filetype=filetype,
                            **kwargs_load if kwargs_load is not None else {},
                        )
                    else:
                        result = func(*args, **kwargs)
                case "overwrite":
                    result = func(*args, **kwargs)
                    _save(
                        result,
                        filepath=filepath,
                        filetype=filetype,
                        **kwargs_save if kwargs_save is not None else {},
                    )
                case "delete":
                    if filepath.exists():
                        filepath.unlink()
                    result = func(*args, **kwargs)
                case "ignore":
                    result = func(*args, **kwargs)
                case _:
                    error = f"Provided mode ({mode}) is invalid"
                    raise ValueError(error)
            return result

        return wrapper

    return decorator


def _save(
    result: Any,  # noqa: ANN401
    /,
    *,
    filepath: Path,
    filetype: Filetype | None,
    **kwargs: Any,  # noqa: ANN401
) -> None:
    identifier = filepath.name

    if filetype is None:
        filetype = _infer_filetype_from_result(result)
        filetype_inferred_from_identifier = _infer_filetype_from_identifier(identifier)
        if filetype != filetype_inferred_from_identifier:
            error = (
                f"Filetype (`{filetype}`) inferred from result does not match the "
                f"filetype (`{filetype_inferred_from_identifier}`) inferred from the"
                f" suffix of the identifier (`{identifier}`). This result will be"
                " cached successfully now, but will fail to load from cache when"
                " re-run. To fix this, ensure that the `identifier` ends with one of"
                f" the expected suffixes ({SUFFIXES[filetype]}) or specify a"
                " `filetype` manually."
            )
            warnings.warn(error, stacklevel=3)

    # Ensure that writes are atomic. In some cases, the handler saves the result
    # to a different filepath than provided (e.g. numpy.savez adds the .npz
    # suffix to filenames that don't have it). To work around this, we create a
    # temporary directory, write to a filepath in that directory, then move the
    # file created in that directory to our desired filepath, whatever its name.
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(
        delete=False,
        dir=filepath.parent,
        prefix="tmp.polycache",
    ) as tmp_dir:
        _get_handler(filetype).save(
            result,
            filepath=Path(tmp_dir) / filepath.name,
            **kwargs,
        )

    # There should be only one file here, but I'm not validating that---just
    # using the first file I find.
    for tmp_file in Path(tmp_dir).glob("*"):
        if tmp_file.is_file():
            tmp_file.move(filepath)
            break

    Path(tmp_dir).rmdir()


def _load(filepath: Path, /, *, filetype: Filetype | None, **kwargs: Any) -> Any:  # noqa: ANN401
    identifier = filepath.name
    filetype = filetype or _infer_filetype_from_identifier(identifier)
    if filetype is None:
        error = (
            "Filetype could not be detected based on the"
            f" suffix of the identifier ({identifier})"
        )
        raise ValueError(error)

    return _get_handler(filetype).load(filepath, **kwargs)


def _infer_filetype_from_identifier(identifier: str, /) -> Filetype | None:
    for filetype, expected_suffixes in SUFFIXES.items():
        for suffix in expected_suffixes:
            if identifier.endswith(suffix):
                return filetype
    return None


def _infer_filetype_from_result(result: Any) -> Filetype:  # noqa: ANN401, C901, PLR0911
    if OPTIONAL_PACKAGES["edfio"] and isinstance(result, edfio.Edf):
        return "EDF"
    if OPTIONAL_PACKAGES["mne"]:
        if isinstance(result, mne.io.BaseRaw):
            return "mne.io.Raw"
        if isinstance(result, mne.Epochs):
            return "mne.Epochs"
    if OPTIONAL_PACKAGES["xarray"] and isinstance(result, xr.DataArray | xr.Dataset):
        return "netCDF4"
    if OPTIONAL_PACKAGES["nibabel"] and isinstance(result, SpatialImage):
        return "NIfTI"
    if OPTIONAL_PACKAGES["numpy"]:
        if isinstance(result, np.ndarray):
            return "numpy"
        if isinstance(result, dict) and all(
            isinstance(value, np.ndarray) for value in result.values()
        ):
            return "numpy.z"
    if OPTIONAL_PACKAGES["PIL"] and isinstance(result, Image):
        return "PIL"
    return "pickle"


def _get_handler(filetype: Filetype) -> ModuleType:
    module = filetype.replace(".", "_").lower()
    return importlib.import_module(f"polycache._handlers.{module}")
