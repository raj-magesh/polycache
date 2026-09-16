from __future__ import annotations

from typing import TYPE_CHECKING, Any

import nibabel as nib

if TYPE_CHECKING:
    from pathlib import Path

    from nibabel.filebasedimages import FileBasedImage
    from nibabel.spatialimages import SpatialImage


def save(
    result: SpatialImage,
    /,
    *,
    filepath: Path,
    **kwargs: Any,  # ruff: ignore[any-type]
) -> None:
    nib.loadsave.save(img=result, filename=filepath, **kwargs)


def load(filepath: Path, /, **kwargs: Any) -> FileBasedImage:  # ruff: ignore[any-type]
    return nib.loadsave.load(filename=filepath, **kwargs)
