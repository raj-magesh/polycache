import pickle  # noqa: S403
from typing import TYPE_CHECKING, Any, Self

from ._base import Handler

if TYPE_CHECKING:
    from pathlib import Path


class PickleHandler(Handler):
    def __init__(self: Self) -> None:
        super().__init__()

    @staticmethod
    def save(result: Any, *, path: Path, **kwargs: Any) -> None:  # noqa: ANN401
        with path.open("wb") as f:
            pickle.dump(result, f, **kwargs)

    @staticmethod
    def load(path: Path, **kwargs: Any) -> Any:  # noqa: ANN401
        with path.open("rb") as f:
            return pickle.load(f, **kwargs)  # noqa: S301
