from os import PathLike
from pathlib import Path
import cv2
from cv2.typing import MatLike
import numpy as np
from PIL import Image

from nivo_reader.lib.common import BoundingBox

OPENCV_SUPPORTED_FORMATS = {
    ".bmp",
    ".dib",
    ".gif",
    ".jpeg",
    ".jpg",
    ".jpe",
    ".jp2",
    ".png",
    ".webp",
    ".avif",
    ".pbm",
    ".pgm",
    ".ppm",
    ".pxm",
    ".pnm",
    ".pfm",
    ".sr",
    ".ras",
    ".tiff",
    ".tif",
    ".exr",
    ".hdr",
    ".pic",
}


def read_matlike_image(image_path: PathLike[str] | Path) -> MatLike:
    """
    Load a BGR or grayscale image as a numpy array

    Args:
        image_path (PathLike[str] | Path): Path to the image file
        grayscale (bool, optional): Whether to load the image in grayscale. Defaults to False.

    Returns:
        MatLike: Image in BGR (or grayscale if grayscale=True) format

    Raises:
        ValueError: If the image cannot be loaded
    """
    image_path = Path(image_path)
    maybe_image: MatLike | None = cv2.cvtColor(
        np.array(Image.open(image_path).convert("RGB")), cv2.COLOR_RGB2BGR
    )
    if maybe_image is None:  # pyright: ignore[reportUnnecessaryComparison]
        raise ValueError(f"Could not load image {image_path}")  # pyright: ignore[reportUnreachable]
    return maybe_image


def extract(image: MatLike, bbox: BoundingBox) -> MatLike:
    """
    Extract rectangular region from image.

    Parameters
    ----------
    image : MatLike
        Input image.
    rect : Rect
        Rectangle (x, y, width, height).

    Returns
    -------
    MatLike
        Extracted region.
    """
    return image[
        max(bbox.y, 0) : min(bbox.y + bbox.height, image.shape[0]),
        max(bbox.x, 0) : min(bbox.x + bbox.width, image.shape[1]),
    ]
    # return image[y : y + h, x : x + w]
