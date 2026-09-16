from __future__ import annotations

from typing import TYPE_CHECKING, Any

from PIL import Image

if TYPE_CHECKING:
    from pathlib import Path


def save(result: Image.Image, /, *, filepath: Path, **kwargs: Any) -> None:  # ruff: ignore[any-type]
    result.save(filepath, **kwargs)


def load(filepath: Path, /, **kwargs: Any) -> Image.Image:  # ruff: ignore[any-type]
    return Image.open(filepath, **kwargs)
