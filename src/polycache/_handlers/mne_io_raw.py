from __future__ import annotations

from typing import TYPE_CHECKING, Any

import mne

if TYPE_CHECKING:
    from pathlib import Path


def save(result: mne.io.BaseRaw, /, *, filepath: Path, **kwargs: Any) -> None:  # ruff: ignore[any-type]
    result.save(filepath, **({"split_naming": "bids"} | kwargs))


def load(filepath: Path, /, **kwargs: Any) -> mne.io.BaseRaw:  # ruff: ignore[any-type]
    try:
        return mne.io.read_raw_fif(filepath, **kwargs)
    except FileNotFoundError:
        split = filepath.name.split("_")
        return mne.io.read_raw_fif(
            filepath.parent / "_".join([*split[:-1], "split-01", split[-1]]),
            **kwargs,
        )
