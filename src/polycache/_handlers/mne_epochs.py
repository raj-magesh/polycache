from typing import TYPE_CHECKING, Any

import mne

if TYPE_CHECKING:
    from pathlib import Path


def save(result: mne.BaseEpochs, /, *, filepath: Path, **kwargs: Any) -> None:  # noqa: ANN401
    result.save(filepath, **({"split_naming": "bids"} | kwargs))


def load(filepath: Path, /, **kwargs: Any) -> mne.BaseEpochs:  # noqa: ANN401
    return mne.read_epochs(filepath, **({"preload": False} | kwargs))
