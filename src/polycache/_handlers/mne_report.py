from __future__ import annotations

from typing import TYPE_CHECKING, Any

import mne

if TYPE_CHECKING:
    from pathlib import Path


def save(
    result: mne.Report,
    /,
    *,
    filepath: Path,
    **kwargs: Any,  # ruff: ignore[any-type]
) -> None:
    result.save(filepath, **kwargs)


def load(filepath: Path, /, **kwargs: Any) -> mne.Report:  # ruff: ignore[any-type]
    return mne.open_report(filepath, **kwargs)
