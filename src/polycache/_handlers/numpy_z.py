from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from pathlib import Path

    from numpy.lib.npyio import NpzFile


def save(
    result: dict[str, npt.NDArray[Any]],
    /,
    *,
    filepath: Path,
    **kwargs: Any,  # ruff: ignore[any-type]
) -> None:
    np.savez_compressed(filepath, **(kwargs | result))


def load(filepath: Path, /, **kwargs: Any) -> NpzFile:  # ruff: ignore[any-type]
    return np.load(filepath, **kwargs)
