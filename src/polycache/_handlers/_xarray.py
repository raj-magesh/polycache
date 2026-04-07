import contextlib
from typing import TYPE_CHECKING, Any, Self

import xarray as xr

from ._base import Handler, _is_installed

if TYPE_CHECKING:
    from pathlib import Path


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
            import cf_xarray  # noqa: PLC0415

            with contextlib.suppress(ValueError):
                result = cf_xarray.encode_multi_index_as_compress(result)

        result.to_netcdf(path, **kwargs)

    @staticmethod
    def load(path: Path, **kwargs: Any) -> xr.DataArray | xr.Dataset:
        result = xr.open_dataset(path, **kwargs)

        # if the Dataset has serialized multi-indexes, deserialize them
        if _is_installed("cf_xarray"):
            import cf_xarray  # noqa: PLC0415

            with contextlib.suppress(ValueError):
                result = cf_xarray.decode_compress_to_multi_index(result)

        # if the Dataset has a single variable, convert it to a DataArray
        if len(result.data_vars) == 1:
            return result[next(iter(result.keys()))]

        return result
