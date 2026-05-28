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
    **kwargs: Any,  # noqa: ANN401
) -> None:
    result.to_filename(filepath, **kwargs)


def load(filepath: Path, /, **kwargs: Any) -> FileBasedImage:  # noqa: ANN401
    return nib.load(filepath, **kwargs)
