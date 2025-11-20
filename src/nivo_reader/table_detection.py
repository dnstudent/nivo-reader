"""Table and line detection utilities for NIVO tables.
Parts of the code are taken from MeteoSaver (https://github.com/VUB-HYDR/MeteoSaver). Credit goes to the authors.
"""

from typing import Literal

import cv2
import numpy as np
from cv2.typing import MatLike, Rect
from scipy.signal import find_peaks

from .configuration.preprocessing import (
    ThresholdParameters,
)
from .configuration.table_and_cell_detection import (
    LinesCombinationParameters,
    LinesDetectionParameters,
    LinesExtractionParameters,
)
from .image_processing import ms_threshold


def ok_side(x: int, expected_x: int, tol: float = 0.1) -> bool:
    """Check if dimension is within tolerance of expected value.

    Args:
        x: Actual dimension
        expected_x: Expected dimension
        tol: Tolerance as fraction (default 0.1 = 10%)

    Returns:
        True if within tolerance
    """
    return (1 - tol) <= (x / expected_x) <= (1 + tol)


def try_detect_table_rect(
    gray_image: MatLike,
    expected_table_width: int,
    expected_table_height: int,
    threshold_parameters: ThresholdParameters,
) -> Rect | None:
    """Detect table rectangle in image.

    Args:
        gray_image: Grayscale image
        expected_table_width: Expected table width
        expected_table_height: Expected table height
        threshold_parameters: Threshold configuration

    Returns:
        Bounding rectangle of table or None if not found
    """
    thresh = ms_threshold(gray_image, threshold_parameters)

    bboxes = sorted(
        filter(
            lambda r: ok_side(r[2], expected_table_width)
            and ok_side(r[3], expected_table_height),
            map(
                lambda cnt: cv2.boundingRect(cnt),
                cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0],
            ),
        ),
        key=lambda r: abs(expected_table_width * expected_table_height - r[2] * r[3]),
    )

    if len(bboxes) > 0:
        return bboxes[0]
    else:
        return None


def cut_out_tables(
    image: MatLike, table_rect: Rect, clip_specs: tuple[int, int, int, int]
) -> tuple[MatLike, MatLike]:
    """Cut out table region from image with optional clipping.

    Args:
        image: Input image
        table_rect: Rectangle defining table bounds
        clip_specs: (up, down, left, right) pixels to clip from table

    Returns:
        tuple of (full_table, clipped_table)
    """
    clip_up, clip_down, clip_left, clip_right = clip_specs
    x, y, w, h = table_rect
    return image[y : y + h, x : x + w], image[
        y + clip_up : y + h - clip_down, x + clip_left : x + w - clip_right
    ]


def detect_lines(
    img_bin: MatLike,
    table_side: int | None,
    parameters: LinesDetectionParameters,
    kind: Literal["vertical", "horizontal"],
) -> MatLike:
    """Detect lines (horizontal or vertical) in binary image.

    Args:
        img_bin: Binary image with white foreground
        table_side: Size of table along relevant axis
        parameters: Line detection configuration
        kind: "vertical" or "horizontal"

    Returns:
        Image with detected lines
    """
    assert kind in ["vertical", "horizontal"]
    if table_side is None:
        kernel_size = (
            img_bin.shape[0] if kind == "horizontal" else img_bin.shape[1]
        ) // parameters.kernel_divisor
    else:
        kernel_size = table_side // parameters.kernel_divisor
    kernel_shape = (1, kernel_size) if kind == "vertical" else (kernel_size, 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_shape)
    eroded_image = cv2.erode(
        img_bin,
        kernel,
        iterations=parameters.erosion_iterations,
    )
    return cv2.dilate(
        eroded_image,
        kernel,
        iterations=parameters.dilation_iterations,
    )


def combine_lines(
    vertical_lines: MatLike,
    horizontal_lines: MatLike,
    parameters: LinesCombinationParameters,
) -> MatLike:
    """Combine vertical and horizontal lines with dilation.

    Args:
        vertical_lines: Vertical line image
        horizontal_lines: Horizontal line image
        parameters: Combination configuration

    Returns:
        Combined and dilated line image
    """
    combined = cv2.addWeighted(vertical_lines, 1, horizontal_lines, 1, 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, parameters.dilation_kernel_shape)
    combined_dilated = cv2.dilate(
        combined, kernel, iterations=parameters.dilation_iterations
    )
    return combined_dilated


def extract_table_lines(
    img_bin: MatLike,
    table_width: int | None,
    table_height: int | None,
    parameters: LinesExtractionParameters,
) -> MatLike:
    """Extract all table lines (horizontal and vertical).

    Args:
        img_bin: Binary image
        table_width: Table width
        table_height: Table height
        parameters: Line extraction configuration

    Returns:
        Image with all table lines
    """
    vertical_lines = detect_lines(
        img_bin,
        table_height,
        parameters=parameters.vertical_lines,
        kind="vertical",
    )
    horizontal_lines = detect_lines(
        img_bin,
        table_width,
        parameters=parameters.horizontal_lines,
        kind="horizontal",
    )
    return combine_lines(
        vertical_lines,
        horizontal_lines,
        parameters=parameters.lines_combination,
    )


def remove_lines_from_image(
    img_bin: MatLike,
    parameters: LinesExtractionParameters,
    table_width: int | None = None,
    table_height: int | None = None,
) -> MatLike:
    """Remove table lines from image.

    Args:
        img_bin: Binary image
        parameters: Line extraction configuration
        table_width: Table width (optional)
        table_height: Table height (optional)

    Returns:
        Image with lines removed
    """
    table_lines = extract_table_lines(img_bin, table_width, table_height, parameters)
    image_wo_lines = cv2.subtract(img_bin, table_lines)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    image_cleaned = cv2.erode(image_wo_lines, kernel, iterations=1)
    return cv2.dilate(image_cleaned, kernel, iterations=1)


def create_word_blobs(image_cleaned: MatLike, parameters) -> MatLike:
    """Convert words to blobs through dilation.

    Args:
        image_cleaned: Cleaned image
        parameters: Word blob creation configuration

    Returns:
        Image with word blobs
    """
    gap_kernel = cv2.getStructuringElement(
        parameters.gap_kernel_type, parameters.gap_kernel_shape
    )
    image_with_blobs = cv2.dilate(
        image_cleaned, gap_kernel, iterations=parameters.gap_iterations
    )

    simple_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, parameters.simple_kernel_shape
    )
    image_with_blobs = cv2.dilate(
        image_with_blobs, simple_kernel, iterations=parameters.simple_iterations
    )
    return image_with_blobs


def detect_column_separators(table_img: MatLike, char_width: int) -> np.ndarray:
    """Detect vertical column separator positions.

    Args:
        table_img: Table image
        char_width: Character width for distance estimation

    Returns:
        Array of column x-coordinates
    """
    vertical_lines = detect_lines(
        table_img,
        table_img.shape[1],
        LinesDetectionParameters(20),
        kind="vertical",
    )
    return find_peaks(
        vertical_lines.sum(axis=0),
        height=table_img.shape[1] * 0.7,
        distance=3 * char_width,
    )[0]


def detect_rows_positions(
    binarized_table_wo_lines: MatLike,
    nchars_threshold: int,
    number_char_shape: tuple[int, int],
) -> np.ndarray:
    """Detect horizontal row positions.

    Args:
        binarized_table_wo_lines: Binary image without lines
        nchars_threshold: Minimum number of characters for peak detection
        number_char_shape: (width, height) of character

    Returns:
        Array of row y-coordinates
    """
    is_white = binarized_table_wo_lines / 255
    n_chars = is_white.sum(axis=1) / number_char_shape[0]
    rows_centers, _ = find_peaks(
        n_chars, height=nchars_threshold, distance=number_char_shape[1] * 0.8
    )
    return rows_centers
