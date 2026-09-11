from __future__ import annotations

from typing import TYPE_CHECKING, Any

import mne

if TYPE_CHECKING:
    from pathlib import Path


def save(
    result: mne.time_frequency.BaseTFR,
    /,
    *,
    filepath: Path,
    **kwargs: Any,  # ruff: ignore[any-type]
) -> None:
    result.save(filepath, **kwargs)


def load(filepath: Path, /, **kwargs: Any) -> mne.time_frequency.BaseTFR:  # ruff: ignore[any-type]
    return mne.time_frequency.read_tfrs(filepath, **kwargs)
