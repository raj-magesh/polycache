from typing import TYPE_CHECKING, Any, Self

import numpy as np
import numpy.typing as npt

from ._base import Handler

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path


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
