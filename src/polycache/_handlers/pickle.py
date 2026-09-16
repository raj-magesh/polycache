from __future__ import annotations

import pickle  # ruff: ignore[suspicious-pickle-import]
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path


def save(result: Any, /, *, filepath: Path, **kwargs: Any) -> None:  # ruff: ignore[any-type]
    with filepath.open("wb") as f:
        pickle.dump(result, f, **kwargs)


def load(filepath: Path, /, **kwargs: Any) -> Any:  # ruff: ignore[any-type]
    with filepath.open("rb") as f:
        return pickle.load(f, **kwargs)  # ruff: ignore[suspicious-pickle-usage]
