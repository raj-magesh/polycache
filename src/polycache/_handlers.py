import contextlib
import importlib
import pickle  # noqa: S403
from abc import ABC, abstractmethod
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Self


def _is_installed(package_name: str) -> bool:
    return importlib.util.find_spec(package_name) is not None


class Handler(ABC):
    def __init__(self: Self) -> None:
        return

    @abstractmethod
    def save(self: Self, result: Any, *, path: Path, **kwargs: Any) -> None:
        pass

    @abstractmethod
    def load(self: Self, path: Path, **kwargs: Any) -> Any:
        pass


class PickleHandler(Handler):
    def __init__(self: Self) -> None:
        super().__init__()

    @staticmethod
    def save(result: Any, *, path: Path, **kwargs) -> None:
        with path.open("wb") as f:
            pickle.dump(result, f, **kwargs)

    @staticmethod
    def load(path: Path, **kwargs: Any) -> Any:
        with path.open("rb") as f:
            return pickle.load(f, **kwargs)


if _is_installed("numpy"):
    import numpy as np
    import numpy.typing as npt

    class NumpyHandler(Handler):
        def __init__(self: Self) -> None:
            super().__init__()

        @staticmethod
        def save(
            result: npt.NDArray[Any] | Mapping[str, npt.NDArray[Any]],
            *,
            path: Path,
            compress: bool = False,
            **kwargs: Any,
        ) -> None:
            if isinstance(result, np.ndarray):
                np.save(path, result, **kwargs)
            elif isinstance(result, dict):
                if compress:
                    np.savez_compressed(path, **result)
                else:
                    np.savez(path, **result)

        @staticmethod
        def load(
            path: Path,
            **kwargs: Any,
        ) -> npt.NDArray[Any] | Mapping[str, npt.NDArray[Any]]:
            return np.load(path, **kwargs)


if _is_installed("xarray"):
    import xarray as xr

    class XarrayHandler(Handler):
        def __init__(self: Self) -> None:
            super().__init__()

        @staticmethod
        def save(
            result: xr.DataArray | xr.Dataset,
            *,
            path: Path,
            **kwargs: Any,
        ) -> None:
            if isinstance(result, xr.DataArray):
                result = result.to_dataset()

            # if the Dataset has multi-indexes, serialize them
            if _is_installed("cf_xarray"):
                import cf_xarray

                with contextlib.suppress(ValueError):
                    result = cf_xarray.encode_multi_index_as_compress(result)

            result.to_netcdf(path, **kwargs)

        @staticmethod
        def load(path: Path, **kwargs: Any) -> xr.DataArray | xr.Dataset:
            result = xr.open_dataset(path, **kwargs)

            # if the Dataset has serialized multi-indexes, deserialize them
            if _is_installed("cf_xarray"):
                import cf_xarray

                with contextlib.suppress(ValueError):
                    result = cf_xarray.decode_compress_to_multi_index(result)

            # if the Dataset has a single variable, convert it to a DataArray
            if len(result.data_vars) == 1:
                return result[next(iter(result.keys()))]

            return result


if _is_installed("nibabel"):
    import nibabel as nib

    class Nifti1ImageHandler(Handler):
        def __init__(self: Self) -> None:
            super().__init__()

        @staticmethod
        def save(
            result: nib.nifti1.Nifti1Image,
            *,
            path: Path,
            **kwargs: Any,
        ) -> None:
            if isinstance(result, nib.nifti1.Nifti1Image):
                nib.save(result, path)

        @staticmethod
        def load(path: Path, **kwargs: Any) -> nib.nifti1.Nifti1Image:
            return nib.load(path)


if _is_installed("PIL"):
    from PIL import Image

    class PillowImageHandler(Handler):
        def __init__(self: Self) -> None:
            super().__init__()

        @staticmethod
        def save(result: Image.Image, *, path: Path, **kwargs: Any) -> None:
            result.save(path, **kwargs)

        @staticmethod
        def load(path: Path, **kwargs: Any) -> Image.Image:
            return Image.open(path, **kwargs)


if _is_installed("edfio"):
    import edfio

    class EdfHandler(Handler):
        def __init__(self: Self) -> None:
            super().__init__()

        @staticmethod
        def save(result: edfio.Edf, *, path: Path, **kwargs: Any) -> None:
            result.write(path, **kwargs)

        @staticmethod
        def load(path: Path, **kwargs: Any) -> edfio.Edf:
            return edfio.read_edf(path, **kwargs)


if _is_installed("mne"):
    import mne

    class MneRawHandler(Handler):
        def __init__(self: Self) -> None:
            super().__init__()

        @staticmethod
        def save(result: mne.io.Raw, *, path: Path, **kwargs: Any) -> None:
            result.save(path, **({"split_naming": "bids"} | kwargs))

        @staticmethod
        def load(path: Path, **kwargs: Any) -> Image.Image:
            return mne.read_raw_fif(path, **kwargs)

    class MneEpochsHandler(Handler):
        def __init__(self: Self) -> None:
            super().__init__()

        @staticmethod
        def save(result: mne.Epochs, *, path: Path, **kwargs: Any) -> None:
            result.save(path, **({"split_naming": "bids"} | kwargs))

        @staticmethod
        def load(path: Path, **kwargs: Any) -> Image.Image:
            return mne.read_epochs(path, **({"preload": False} | kwargs))


def get_handler(filetype: str) -> Handler:
    match filetype:
        case "pickle":
            return PickleHandler()
        case "numpy":
            if _is_installed("numpy"):
                return NumpyHandler()
        case "netCDF4":
            if _is_installed("xarray"):
                return XarrayHandler()
        case "NIfTI":
            if _is_installed("nibabel"):
                return Nifti1ImageHandler()
        case "PIL":
            if _is_installed("PIL"):
                return PillowImageHandler()
        case "EDF":
            if _is_installed("edfio"):
                return EdfHandler()
        case "mne.io.Raw":
            if _is_installed("mne"):
                return MneEpochsHandler()
        case "mne.Epochs":
            if _is_installed("mne"):
                return MneEpochsHandler()

    error = f"Handler for filetype {filetype} not supported"
    raise ValueError(error)
