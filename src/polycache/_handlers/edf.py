from __future__ import annotations

from typing import TYPE_CHECKING, Any

import edfio

if TYPE_CHECKING:
    from pathlib import Path


def save(result: edfio.Edf, /, *, filepath: Path, **kwargs: Any) -> None:  # ruff: ignore[any-type]
    result.write(filepath, **kwargs)


def load(filepath: Path, /, **kwargs: Any) -> edfio.Edf:  # ruff: ignore[any-type]
    return edfio.read_edf(filepath, **kwargs)
