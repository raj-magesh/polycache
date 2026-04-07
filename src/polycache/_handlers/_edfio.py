from typing import TYPE_CHECKING, Any, Self

import edfio

from ._base import Handler

if TYPE_CHECKING:
    from pathlib import Path


class EdfHandler(Handler):
    def __init__(self: Self) -> None:
        super().__init__()

    @staticmethod
    def save(result: edfio.Edf, *, path: Path, **kwargs: Any) -> None:
        result.write(path, **kwargs)

    @staticmethod
    def load(path: Path, **kwargs: Any) -> edfio.Edf:
        return edfio.read_edf(path, **kwargs)
