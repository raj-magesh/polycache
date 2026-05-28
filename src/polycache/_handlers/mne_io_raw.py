from typing import TYPE_CHECKING, Any

import mne

if TYPE_CHECKING:
    from pathlib import Path


def save(result: mne.io.BaseRaw, /, *, filepath: Path, **kwargs: Any) -> None:  # noqa: ANN401
    result.save(filepath, **({"split_naming": "bids"} | kwargs))


def load(filepath: Path, /, **kwargs: Any) -> mne.io.BaseRaw:  # noqa: ANN401
    return mne.io.read_raw_fif(filepath, **kwargs)
