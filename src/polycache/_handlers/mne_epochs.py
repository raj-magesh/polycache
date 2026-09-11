from __future__ import annotations

from typing import TYPE_CHECKING, Any

import mne

if TYPE_CHECKING:
    from pathlib import Path


def save(result: mne.BaseEpochs, /, *, filepath: Path, **kwargs: Any) -> None:  # ruff: ignore[any-type]
    result.save(filepath, **({"split_naming": "bids"} | kwargs))


def load(filepath: Path, /, **kwargs: Any) -> mne.BaseEpochs:  # ruff: ignore[any-type]
    try:
        return mne.read_epochs(filepath, **({"preload": False} | kwargs))
    except FileNotFoundError:
        split = filepath.name.split("_")
        return mne.read_epochs(
            filepath.parent / "_".join([*split[:-1], "split-01", split[-1]]),
            **kwargs,
        )
