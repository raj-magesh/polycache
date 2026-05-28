from typing import TYPE_CHECKING, Any

from PIL import Image

if TYPE_CHECKING:
    from pathlib import Path


def save(result: Image.Image, /, *, filepath: Path, **kwargs: Any) -> None:  # noqa: ANN401
    result.save(filepath, **kwargs)


def load(filepath: Path, /, **kwargs: Any) -> Image.Image:  # noqa: ANN401
    return Image.open(filepath, **kwargs)
