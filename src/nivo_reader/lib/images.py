from os import PathLike
from pathlib import Path
import cv2
from cv2.typing import MatLike

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


def read_matlike_image(
    image_path: PathLike[str] | Path, grayscale: bool = False
) -> MatLike:
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
    flags = cv2.IMREAD_COLOR if not grayscale else cv2.IMREAD_GRAYSCALE
    maybe_image: MatLike | None = cv2.imread(str(image_path), flags)
    if maybe_image is None:
        raise ValueError(f"Could not load image {image_path}")
    return maybe_image
