from typing import TYPE_CHECKING, Any, Self

import mne

from ._base import Handler

if TYPE_CHECKING:
    from pathlib import Path


class MneRawHandler(Handler):
    def __init__(self: Self) -> None:
        super().__init__()

    @staticmethod
    def save(result: mne.io.BaseRaw, *, path: Path, **kwargs: Any) -> None:
        result.save(path, **({"split_naming": "bids"} | kwargs))

    @staticmethod
    def load(path: Path, **kwargs: Any) -> mne.io.BaseRaw:
        return mne.io.read_raw_fif(path, **kwargs)


class MneEpochsHandler(Handler):
    def __init__(self: Self) -> None:
        super().__init__()

    @staticmethod
    def save(result: mne.BaseEpochs, *, path: Path, **kwargs: Any) -> None:
        result.save(path, **({"split_naming": "bids"} | kwargs))

    @staticmethod
    def load(path: Path, **kwargs: Any) -> mne.BaseEpochs:
        return mne.read_epochs(path, **({"preload": False} | kwargs))
