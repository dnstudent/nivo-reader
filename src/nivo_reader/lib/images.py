from os import PathLike
from pathlib import Path
import cv2
from cv2.typing import MatLike
import numpy as np
from PIL import Image

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
