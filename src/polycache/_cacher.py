import functools
import importlib
import importlib.util
import inspect
import os
import tempfile
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

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
    "xarray",
    "NIfTI",
    "numpy",
    "numpy.z",
    "pickle",
    "PIL",
]

Mode = Literal["normal", "readonly", "overwrite", "disabled"]

SUFFIXES: dict[Filetype, tuple[str, ...]] = {
    "EDF": (".edf",),
    "mne.Epochs": ("-epo.fif",),
    "mne.io.Raw": (".fif",),
    "xarray": (".nc",),
    "NIfTI": (".nii.gz", ".nii"),
    "numpy.z": (".npz",),
    "numpy": (".npy",),
    "pickle": (".pkl",),
    "PIL": (".png", ".jpg"),
}


def cache[**P, R](  # noqa: PLR0913
    identifier: str,
    *,
    filetype: Filetype | None = None,
    root_dir: Path | None = None,
    mode: Mode = "normal",
    remapper: Callable[[Mapping[str, Any]], dict[str, str]] | None = None,
    save_kws: Mapping[str, Any] | None = None,
    load_kws: Mapping[str, Any] | None = None,
    save_fn: Callable[[Any, Path], None] | None = None,
    load_fn: Callable[[Path], Any] | None = None,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Cache outputs of function calls to disk.

    When this decorator is applied to a function and the decorated function is called for the first time, its output is stored on disk. Whenever the decorated function is called again, the stored value is retrieved from disk and returned.

    This decorator can be customized extensively to

    - flexibly parameterize the filepath of the saved output based on the function arguments (see ``identifier`` and ``remapper``),
    - use different file formats based on the return type of the function output (see ``filetype``),
    - control the save/load processes (see ``save_kws`` and ``load_kws``),
    - use custom save/load functions to support new file formats (see the ``save_fn`` and ``load_fn`` parameters)
    - change the cache directory (see ``root_dir``), and
    - change how cached outputs are used (see ``mode``).

    Parameters
    ----------
    identifier
        Filepath where the function's out should be cached, relative to ``root_dir``. See the Examples section for how this filepath can be parameterized by the function arguments.
    filetype
        File format used to cache the function's output.

        When ``None`` (default), automatically selects a file format based on the return type of the function. In this case, ``identifier`` must end with a valid suffix (see the documentation); otherwise, cached results cannot be successfully loaded from disk.

        The `pickle <https://docs.python.org/3/library/pickle.html>`_ format is used as a fallback if no more appropriate file format is available.
    root_dir
        Cache directory to save files to/load files from. Defaults to the first of ``$POLYCACHE_HOME``, ``$XDG_CACHE_HOME/polycache``, and ``$HOME/.cache/polycache``.
    mode
        Controls how cached outputs are used:
            - ``"normal"`` (default): If the function's output has been previously cached, the stored value is retrieved and returned. If not, the function is run and its output is cached.
            - ``"readonly"``: If the function's output has been previously cached, the stored value is retrieved and returned. If not, the function is run but its output is NOT cached.
            - ``"overwrite"``: The function is run and its output is cached, overwriting any existing cached output, if it exists.
            - ``"disabled"``: The function is run and its output is returned. Any existing cached values are ignored.
    remapper
        Function supporting more complex parameterization of the identifier by the function arguments. See the Examples section for usage.
    save_kws
        Keyword arguments passed to the save function (see the documentation).
    load_kws
        Keyword arguments passed to the load function (see the documentation).
    save_fn
        Custom function to use to save the result of the function. Accepts exactly two arguments: the function result and the filepath.
    load_fn
        Custom function to use to load the cached result from the filepath. Accepts exactly one argument: the filepath.

    Returns
    -------
    A decorator that can be applied to any function to cache its outputs to disk.


    """  # noqa: E501
    # TODO Investigate if ``remapper`` can be replaced with template strings

    def decorator[**P, R](func: Callable[P, R]) -> Callable[P, R]:
        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:  # noqa: PLR0912
            # temporarily strip annotations because inspecting the signature fails at runtime if the type of the parameter is only defined in a TYPE_CHECKING block
            annotations = getattr(func, "__annotations__", None)
            func.__annotations__ = {}
            try:
                signature = inspect.signature(func)
            finally:
                if annotations is None:
                    del func.__annotations__
                else:
                    func.__annotations__ = annotations

            bound_arguments = signature.bind(*args, **kwargs)
            bound_arguments.apply_defaults()
            args_to_format = bound_arguments.arguments

            if remapper:
                args_to_format = remapper(args_to_format)

            filepath = (
                (root_dir or get_polycache_home())
                / identifier.format(**args_to_format)
            )  # fmt: skip

            saver = save_fn or functools.partial(
                save,
                filetype=filetype,
                **save_kws if save_kws is not None else {},
            )
            loader = load_fn or functools.partial(
                load,
                filetype=filetype,
                **load_kws if load_kws is not None else {},
            )

            match mode:
                case "normal":
                    if filepath.exists():
                        result = loader(filepath)
                    else:
                        result = func(*args, **kwargs)
                        saver(result, filepath)
                case "readonly":
                    if filepath.exists():
                        result = loader(filepath)
                    else:
                        result = func(*args, **kwargs)
                case "overwrite":
                    result = func(*args, **kwargs)
                    saver(result, filepath)
                case "disabled":
                    result = func(*args, **kwargs)
                case _:
                    error = f"Provided mode ({mode}) is invalid"
                    raise ValueError(error)
            return result

        return wrapper

    return decorator


def get_polycache_home() -> Path:
    if polycache_home := os.getenv("POLYCACHE_HOME"):
        return Path(polycache_home)
    if xdg_cache_home := os.getenv("XDG_CACHE_HOME"):
        return Path(xdg_cache_home) / "polycache"
    return Path.home() / ".cache" / "polycache"


def save(
    result: Any,  # noqa: ANN401
    filepath: Path,
    /,
    *,
    filetype: Filetype | None,
    **kwargs: Any,  # noqa: ANN401
) -> None:
    identifier = filepath.name

    if filetype is None:
        filetype = infer_filetype_from_result(result)
        filetype_inferred_from_identifier = infer_filetype_from_identifier(identifier)
        if filetype != filetype_inferred_from_identifier:
            error = f"Filetype (`{filetype}`) inferred from result does not match the filetype (`{filetype_inferred_from_identifier}`) inferred from the suffix of the identifier (`{identifier}`). This result will be cached successfully now, but will fail to load from cache when re-run. To fix this, ensure that the `identifier` ends with one of the expected suffixes ({SUFFIXES[filetype]}) or specify a `filetype` manually."  # noqa: E501
            warnings.warn(error, stacklevel=3)

    # Ensure that writes are atomic. In some cases, the handler saves the result to a different filepath than provided (e.g. numpy.savez adds the .npz suffix to filenames that don't have it). To work around this, we create a temporary directory, write to a filepath in that directory, then move the file created in that directory to our desired filepath, whatever its name.  # noqa: E501
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(
        delete=False,
        dir=filepath.parent,
        prefix="tmp.polycache",
    ) as tmp_dir:
        get_handler(filetype).save(
            result,
            filepath=Path(tmp_dir) / filepath.name,
            **kwargs,
        )

    # There should be only one file here, but I'm not validating that---just using the first file I find.  # noqa: E501
    for tmp_file in Path(tmp_dir).glob("*"):
        if tmp_file.is_file():
            tmp_file.move(filepath)
            break

    Path(tmp_dir).rmdir()


def load(filepath: Path, /, *, filetype: Filetype | None, **kwargs: Any) -> Any:  # noqa: ANN401
    identifier = filepath.name
    filetype = filetype or infer_filetype_from_identifier(identifier)
    if filetype is None:
        error = (
            "Filetype could not be detected based on the"
            f" suffix of the identifier ({identifier})"
        )
        raise ValueError(error)

    return get_handler(filetype).load(filepath, **kwargs)


def infer_filetype_from_identifier(identifier: str, /) -> Filetype | None:
    for filetype, expected_suffixes in SUFFIXES.items():
        for suffix in expected_suffixes:
            if identifier.endswith(suffix):
                return filetype
    return None


def infer_filetype_from_result(result: Any) -> Filetype:  # noqa: ANN401, C901, PLR0911
    if OPTIONAL_PACKAGES["edfio"] and isinstance(result, edfio.Edf):
        return "EDF"
    if OPTIONAL_PACKAGES["mne"]:
        if isinstance(result, mne.io.BaseRaw):
            return "mne.io.Raw"
        if isinstance(result, mne.Epochs):
            return "mne.Epochs"
    if OPTIONAL_PACKAGES["xarray"] and isinstance(result, (xr.DataArray, xr.Dataset)):
        return "xarray"
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


def get_handler(filetype: Filetype) -> ModuleType:
    module = filetype.replace(".", "_").lower()
    return importlib.import_module(f"polycache._handlers.{module}")
