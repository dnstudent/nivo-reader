"""Image preprocessing utilities for NIVO table extraction. Parts of the code are taken from MeteoSaver (https://github.com/VUB-HYDR/MeteoSaver). Credit goes to the authors."""

import cv2
import numpy as np
from cv2.typing import MatLike
from img2table.document.base.rotation import fix_rotation_image

from .original_parameterization.preprocessing import (
    BinarizationParameters,
    PreprocessingParameters,
    ThresholdParameters,
)


def my_table_struct(bw_image: MatLike) -> MatLike:
    """Extract table structure using morphological operations.

    Args:
        bw_image: Binary image where foreground is white

    Returns:
        Image with detected table structure
    """
    op = cv2.MORPH_OPEN
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
    horizontal_lines = cv2.morphologyEx(
        bw_image,
        op,
        horizontal_kernel,
        iterations=5,
    )
    horizontal_lines = cv2.morphologyEx(
        horizontal_lines,
        cv2.MORPH_DILATE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (10, 2)),
        iterations=1,
    )

    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 25))
    vertical_lines = cv2.morphologyEx(bw_image, op, vertical_kernel, iterations=5)
    vertical_lines = cv2.morphologyEx(
        vertical_lines,
        cv2.MORPH_DILATE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (2, 10)),
        iterations=1,
    )
    return cv2.add(horizontal_lines, vertical_lines)


def ms_binary(
    image_in_grayscale: MatLike, binarization_parameters: BinarizationParameters
) -> MatLike:
    """Apply adaptive thresholding to create a binary image.

    Args:
        image_in_grayscale: Grayscale image
        binarization_parameters: Binarization configuration

    Returns:
        Binary image
    """
    binarized_image = cv2.adaptiveThreshold(
        image_in_grayscale,
        255,
        binarization_parameters.adaptive_threshold_type,
        cv2.THRESH_BINARY,
        binarization_parameters.region_side,
        binarization_parameters.threshold_c,
    )
    return binarized_image


def ms_threshold(image: MatLike, threshold_parameters: ThresholdParameters) -> MatLike:
    """Apply Otsu's thresholding and morphological closing.

    Args:
        image: Input image (grayscale or color)
        threshold_parameters: Threshold configuration

    Returns:
        Thresholded image
    """
    if image.ndim > 2:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    kernel = np.ones(threshold_parameters.kernel_shape, np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    return thresh


def preproc(
    image: MatLike, preprocessing_parameters: PreprocessingParameters
) -> tuple[MatLike, MatLike, MatLike, MatLike]:
    """Preprocess image: rotate, convert to grayscale, binarize, threshold.

    Args:
        image: Input image
        preprocessing_parameters: Preprocessing configuration

    Returns:
        tuple of (original, threshold, binarized, table_struct) images
    """
    # Fix rotation
    image = fix_rotation_image(image)[0]

    # Convert to grayscale
    image_in_grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply adaptive thresholding
    binarized_image = ms_binary(
        image_in_grayscale, preprocessing_parameters.binarization_parameters
    )

    # Apply Otsu's thresholding
    thresh = ms_threshold(
        image_in_grayscale, preprocessing_parameters.threshold_parameters
    )

    return (
        image,
        thresh,
        binarized_image,
        my_table_struct(thresh),
    )


def extract_contours_boxes(image: MatLike) -> list:
    """Extract bounding boxes from contours in the image.

    Args:
        image: Processed image

    Returns:
        List of bounding rectangles
    """
    contours = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]
    return [
        cv2.boundingRect(contour)
        for contour in contours
        if cv2.contourArea(contour) > 4
    ]
