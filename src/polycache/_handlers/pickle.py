import pickle  # noqa: S403
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path


def save(result: Any, /, *, filepath: Path, **kwargs: Any) -> None:  # noqa: ANN401
    with filepath.open("wb") as f:
        pickle.dump(result, f, **kwargs)


def load(filepath: Path, /, **kwargs: Any) -> Any:  # noqa: ANN401
    with filepath.open("rb") as f:
        return pickle.load(f, **kwargs)  # noqa: S301
