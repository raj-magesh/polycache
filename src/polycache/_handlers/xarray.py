from __future__ import annotations

import contextlib
import importlib.util
from typing import TYPE_CHECKING, Any

import xarray as xr

if TYPE_CHECKING:
    from pathlib import Path


def save(
    result: xr.DataArray | xr.Dataset,
    /,
    *,
    filepath: Path,
    **kwargs: Any,  # noqa: ANN401
) -> None:
    if isinstance(result, xr.DataArray):
        result = result.to_dataset()

    # if the Dataset has multi-indexes, serialize them
    if importlib.util.find_spec("cf_xarray"):
        import cf_xarray  # noqa: PLC0415

        with contextlib.suppress(ValueError):
            result = cf_xarray.encode_multi_index_as_compress(result)

    result.to_netcdf(filepath, **kwargs)


def load(filepath: Path, /, **kwargs: Any) -> xr.DataArray | xr.Dataset:  # noqa: ANN401
    result = xr.open_dataset(filepath, **kwargs)

    # if the Dataset has serialized multi-indexes, deserialize them
    if importlib.util.find_spec("cf_xarray"):
        import cf_xarray  # noqa: PLC0415

        with contextlib.suppress(ValueError):
            result = cf_xarray.decode_compress_to_multi_index(result)

    # if the Dataset has a single variable, convert it to a DataArray
    if len(result.data_vars) == 1:
        return result[next(iter(result.keys()))]

    return result
