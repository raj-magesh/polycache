from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from pathlib import Path


def save(result: npt.NDArray[Any], /, *, filepath: Path, **kwargs: Any) -> None:  # noqa: ANN401
    np.save(filepath, result, **kwargs)


def load(filepath: Path, /, **kwargs: Any) -> npt.NDArray[Any]:  # noqa: ANN401
    return np.load(filepath, **kwargs)
