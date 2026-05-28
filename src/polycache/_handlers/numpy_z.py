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
    compress: bool = False,
    **kwargs: Any,  # noqa: ANN401
) -> None:
    if compress:
        np.savez_compressed(filepath, **(kwargs | result))
    else:
        np.savez(filepath, **(kwargs | result))


def load(filepath: Path, /, **kwargs: Any) -> NpzFile:  # noqa: ANN401
    return np.load(filepath, **kwargs)
