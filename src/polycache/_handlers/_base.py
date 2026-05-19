import importlib.util
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Self

if TYPE_CHECKING:
    from pathlib import Path


def _is_installed(package_name: str) -> bool:
    return importlib.util.find_spec(package_name) is not None


class Handler(ABC):
    def __init__(self: Self) -> None:
        return

    @abstractmethod
    def save(result: Any, *, path: Path, **kwargs: Any) -> None:
        pass

    @abstractmethod
    def load(path: Path, **kwargs: Any) -> Any:
        pass


def get_handler(filetype: str) -> Handler:
    match filetype:
        case "pickle":
            from ._pickle import PickleHandler  # noqa: PLC0415

            return PickleHandler()
        case "numpy":
            if _is_installed("numpy"):
                from ._numpy import NumpyHandler  # noqa: PLC0415

                return NumpyHandler()
        case "netCDF4":
            if _is_installed("xarray"):
                from ._xarray import XarrayHandler  # noqa: PLC0415

                return XarrayHandler()
        case "NIfTI":
            if _is_installed("nibabel"):
                from ._nibabel import NiftiSpatialImageHandler  # noqa: PLC0415

                return NiftiSpatialImageHandler()
        case "PIL":
            if _is_installed("PIL"):
                from ._pil import PillowImageHandler  # noqa: PLC0415

                return PillowImageHandler()
        case "EDF":
            if _is_installed("edfio"):
                from ._edfio import EdfHandler  # noqa: PLC0415

                return EdfHandler()
        case "mne.io.Raw":
            if _is_installed("mne"):
                from ._mne import MneRawHandler  # noqa: PLC0415

                return MneRawHandler()
        case "mne.Epochs":
            if _is_installed("mne"):
                from ._mne import MneEpochsHandler  # noqa: PLC0415

                return MneEpochsHandler()

    error = f"Handler for filetype {filetype} not supported"
    raise ValueError(error)
