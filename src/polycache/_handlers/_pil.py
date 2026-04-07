from typing import TYPE_CHECKING, Any, Self

from PIL import Image

from ._base import Handler

if TYPE_CHECKING:
    from pathlib import Path


class PillowImageHandler(Handler):
    def __init__(self: Self) -> None:
        super().__init__()

    @staticmethod
    def save(result: Image.Image, *, path: Path, **kwargs: Any) -> None:
        result.save(path, **kwargs)

    @staticmethod
    def load(path: Path, **kwargs: Any) -> Image.Image:
        return Image.open(path, **kwargs)
