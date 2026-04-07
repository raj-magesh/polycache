from typing import TYPE_CHECKING, Any, Self

import nibabel as nib

from ._base import Handler

if TYPE_CHECKING:
    from pathlib import Path

    from nibabel.filebasedimages import FileBasedImage
    from nibabel.spatialimages import SpatialImage


class NiftiSpatialImageHandler(Handler):
    def __init__(self: Self) -> None:
        super().__init__()

    @staticmethod
    def save(
        result: SpatialImage,
        *,
        path: Path,
        **kwargs: Any,
    ) -> None:
        result.to_filename(path)

    @staticmethod
    def load(path: Path, **kwargs: Any) -> FileBasedImage:
        return nib.load(path)
