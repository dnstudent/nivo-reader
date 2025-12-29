"""Table and line detection utilities for NIVO tables.
Parts of the code are taken from MeteoSaver (https://github.com/VUB-HYDR/MeteoSaver). Credit goes to the authors.
"""

import cv2
import numpy as np
from cv2.typing import MatLike, Rect
from scipy.signal import find_peaks

from .configuration.preprocessing import (
    ThresholdParameters,
    LinesDetectionParameters,
)
from .configuration.table_and_cell_detection import (
    LinesExtractionParameters,
    WordBlobsCreationParameters,
)
from .image_processing import ms_threshold, detect_lines, combine_lines


def ok_side(x: int, expected_x: int, tol: float) -> bool:
    """
    Check if dimension is within tolerance of expected value.

    Parameters
    ----------
    x : int
        Actual dimension.
    expected_x : int
        Expected dimension.
    tol : float, optional
        Tolerance as fraction (default 0.1 = 10% is hardcoded in usage).

    Returns
    -------
    bool
        True if within tolerance.
    """
    return (1 - tol) <= (x / expected_x) <= (1 + tol)


def try_detect_table_rect(
    gray_image: MatLike,
    expected_table_shape: tuple[int, int],
    threshold_parameters: ThresholdParameters,
) -> Rect | None:
    """
    Detect table rectangle in image.

    Parameters
    ----------
    gray_image : MatLike
        Grayscale image.
    expected_table_shape : tuple[int, int]
        Expected table (width, height).
    threshold_parameters : ThresholdParameters
        Threshold configuration.

    Returns
    -------
    Rect | None
        Bounding rectangle of table or None if not found.
    """
    thresh = ms_threshold(gray_image, threshold_parameters)
    expected_table_width, expected_table_height = expected_table_shape
    bboxes = sorted(
        filter(
            # TODO: explain tol are fixed, why hardcoded?
            lambda r: ok_side(r[2], expected_table_width, tol=0.1)
            and ok_side(r[3], expected_table_height, tol=0.1),
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
    """
    Cut out table region from image with optional clipping.

    Parameters
    ----------
    image : MatLike
        Input image.
    table_rect : Rect
        Rectangle defining table bounds.
    clip_specs : tuple[int, int, int, int]
        (up, down, left, right) pixels to clip from table.

    Returns
    -------
    tuple[MatLike, MatLike]
        Tuple of (full_table, clipped_table).
    """
    clip_up, clip_down, clip_left, clip_right = clip_specs
    x, y, w, h = table_rect
    return image[y : y + h, x : x + w], image[
        y + clip_up : y + h - clip_down, x + clip_left : x + w - clip_right
    ]


def extract_table_lines(
    img_bin: MatLike,
    table_width: int | None,
    table_height: int | None,
    parameters: LinesExtractionParameters,
) -> MatLike:
    """
    Extract all table lines (horizontal and vertical).

    Parameters
    ----------
    img_bin : MatLike
        Binary image.
    table_width : int | None
        Table width.
    table_height : int | None
        Table height.
    parameters : LinesExtractionParameters
        Line extraction configuration.

    Returns
    -------
    MatLike
        Image with all table lines.
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
    """
    Remove table lines from image.

    Parameters
    ----------
    img_bin : MatLike
        Binary image.
    parameters : LinesExtractionParameters
        Line extraction configuration.
    table_width : int | None, optional
        Table width.
    table_height : int | None, optional
        Table height.

    Returns
    -------
    MatLike
        Image with lines removed.
    """
    table_lines = extract_table_lines(img_bin, table_width, table_height, parameters)
    image_wo_lines = cv2.subtract(img_bin, table_lines)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    image_cleaned = cv2.erode(image_wo_lines, kernel, iterations=1)
    return cv2.dilate(image_cleaned, kernel, iterations=1)


def create_word_blobs(
    image_cleaned: MatLike, parameters: WordBlobsCreationParameters
) -> MatLike:
    """
    Convert words to blobs through dilation.

    Parameters
    ----------
    image_cleaned : MatLike
        Cleaned image.
    parameters : WordBlobsCreationParameters
        Word blob creation configuration.

    Returns
    -------
    MatLike
        Image with word blobs.
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


def detect_column_separators(table_img: MatLike, char_width: int) -> list[int]:
    """
    Detect vertical column separator positions.

    Parameters
    ----------
    table_img : MatLike
        Table image.
    char_width : int
        Character width for distance estimation.

    Returns
    -------
    list[int]
        List of column x-coordinates.
    """
    vertical_lines = detect_lines(
        table_img,
        table_img.shape[1],
        LinesDetectionParameters(20),
        kind="vertical",
    )
    peaks: list[int] = list(
        find_peaks(
            vertical_lines.sum(axis=0),
            height=table_img.shape[1] * 0.7,
            distance=3 * char_width,
        )[0]
    )
    if peaks[0] > 2 * char_width:
        peaks.insert(0, 0)
    return peaks


def detect_rows_positions(
    binarized_table_wo_lines: MatLike,
    nchars_threshold: int,
    number_char_shape: tuple[int, int],
) -> np.ndarray:
    """
    Detect horizontal row positions.

    Parameters
    ----------
    binarized_table_wo_lines : MatLike
        Binary image without lines.
    nchars_threshold : int
        Minimum number of characters for peak detection.
    number_char_shape : tuple[int, int]
        (width, height) of character.

    Returns
    -------
    np.ndarray
        Array of row y-coordinates.
    """
    is_white = binarized_table_wo_lines / 255
    n_chars = is_white.sum(axis=1) / number_char_shape[0]
    rows_centers, _ = find_peaks(
        n_chars,
        height=nchars_threshold,
        distance=number_char_shape[1] * 0.8,  # TODO: explain 0.8 factor
    )
    return rows_centers
