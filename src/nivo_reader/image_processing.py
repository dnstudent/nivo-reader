"""Image preprocessing utilities for NIVO table extraction. Parts of the code are taken from MeteoSaver (https://github.com/VUB-HYDR/MeteoSaver). Credit goes to the authors."""

from typing import Literal
import logging

import cv2
from cv2.typing import MatLike, Rect
import numpy as np
from img2table.document.base.rotation import fix_rotation_image
from img2table.document.base.rotation import (
    get_relevant_angles,
    estimate_skew,
    get_connected_components,
)

from .configuration.preprocessing import (
    BinarizationConfiguration,
    PreprocessConfiguration,
    ThresholdConfiguration,
    LinesDetectionConfiguration,
    LinesCombinationConfiguration,
    AngleDetectionConfiguration,
)

logger = logging.getLogger("nivo_reader.image_processing")


def detect_lines(
    img_bin: MatLike,
    table_side: int | None,
    kind: Literal["vertical", "horizontal"],
    configuration: LinesDetectionConfiguration,
) -> MatLike:
    """
    Detect lines (horizontal or vertical) in binary image.

    Parameters
    ----------
    img_bin : MatLike
        Binary image with white foreground.
    table_side : int | None
        Size of table along relevant axis.
    configuration : LinesDetectionConfiguration
        Line detection configuration.
    kind : Literal["vertical", "horizontal"]
        "vertical" or "horizontal".

    Returns
    -------
    MatLike
        Image with detected lines.
    """
    assert kind in ["vertical", "horizontal"]
    if img_bin.mean() > 127.5:
        logger.warning(
            "The image passed to line detection has more foreground color than background. Did you forget to invert it?"
        )
    if table_side is None:
        kernel_size = (
            img_bin.shape[0] if kind == "horizontal" else img_bin.shape[1]
        ) // configuration.kernel_divisor
    else:
        kernel_size = table_side // configuration.kernel_divisor
    kernel_shape = (1, kernel_size) if kind == "vertical" else (kernel_size, 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_shape)
    eroded_image = cv2.erode(
        img_bin,
        kernel,
        kernel,
        iterations=configuration.erosion_iterations,
    )
    return cv2.dilate(
        eroded_image,
        kernel,
        iterations=configuration.dilation_iterations,
    )


def combine_lines(
    vertical_lines: MatLike,
    horizontal_lines: MatLike,
    configuration: LinesCombinationConfiguration,
) -> MatLike:
    """
    Combine vertical and horizontal lines with dilation.

    Parameters
    ----------
    vertical_lines : MatLike
        Vertical line image.
    horizontal_lines : MatLike
        Horizontal line image.
    configuration : LinesCombinationConfiguration
        Combination configuration.

    Returns
    -------
    MatLike
        Combined and dilated line image.
    """
    combined = cv2.addWeighted(vertical_lines, 1, horizontal_lines, 1, 1)
    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, configuration.dilation_kernel_shape
    )
    combined_dilated = cv2.dilate(
        combined, kernel, iterations=configuration.dilation_iterations
    )
    return combined_dilated


def contour_hw_ratio(contour: MatLike) -> float:
    """
    Calculate the height-to-width ratio of a contour.

    Parameters
    ----------
    contour : MatLike
        Input contour.

    Returns
    -------
    float
        Height/Width ratio.
    """
    _, _, w, h = cv2.boundingRect(contour)
    return h / w


# Inspired by MeteoSaver
def nivo_lines_angle(
    image: MatLike,
    orientation: Literal["vertical", "horizontal"],
    configuration: AngleDetectionConfiguration,
) -> float | None:
    """
    Calculate the median angle of lines in the specified orientation.

    Parameters
    ----------
    image : MatLike
        Input image.
    orientation : Literal["vertical", "horizontal"]
        Orientation of lines to detect ("vertical" or "horizontal").
    lines_detection_configuration : LinesDetectionConfiguration
        Configuration for line detection.

    Returns
    -------
    float | None
        Median angle of detected lines in degrees, or None if no valid lines found.
    """
    vertical_lines = detect_lines(
        image,
        table_side=None,
        kind=orientation,
        configuration=configuration.lines_detection_configuration,
    )

    vertical_boxes = list(
        filter(
            lambda box: box[2] > 0
            and box[3] / box[2] > configuration.line_ratio_threshold,
            extract_contours_boxes(vertical_lines),
        )
    )

    # Compute the angle of each line as the arctangent of the height-to-width ratio
    angles = [np.arctan(h / w) if w > 1 else np.pi / 2 for _, _, w, h in vertical_boxes]

    if angles:
        return float(np.degrees(np.median(angles)))
    else:
        return None


# TODO: Improve attribution
# Taken from img2table. Credit goes to the authors.
def img2table_lines_angle(img: MatLike) -> float:
    """
    Fix rotation of input image (based on https://www.mdpi.com/2079-9292/9/1/55) by at most 45 degrees.

    Parameters
    ----------
    img : MatLike
        Image array.

    Returns
    -------
    float
        Estimated skew angle.
    """
    # Get connected components of the images
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    cc_centroids, ref_height, thresh = get_connected_components(img=gray)

    # Check number of centroids
    if len(cc_centroids) < 2:
        return 0.0

    # Compute most likely angles from connected components
    angles = get_relevant_angles(centroids=cc_centroids, ref_height=ref_height)
    # Estimate skew
    return estimate_skew(angles=angles, thresh=thresh)


def deskew_nivo(
    wb_table_image: MatLike,
    *oth_images: *tuple[MatLike, ...],
    configuration: AngleDetectionConfiguration,
):
    """
    Deskews an image by detecting and correcting its skew based on the orientation of detected horizontal lines.

    This function corrects the skew of an input image by first detecting horizontal lines within the image using morphological operations. It calculates the average angle of these detected lines and rotates the image by this angle to align the horizontal lines correctly, effectively deskewing the image. The result is an image where the content is horizontally aligned, which is particularly useful for preprocessing before further analysis or OCR (Optical Character Recognition).

    Parameters
    ----------
    wb_table_image : MatLike
        The input image that needs to be deskewed. This image must be binary: white foreground and black background. It should be just the table.
    *oth_images : MatLike
        Other images to rotate by the same angle.
    configuration : AngleDetectionConfiguration
        Configuration for angle detection.

    Returns
    -------
    float | None
        The median angle of vertical lines.
    MatLike
        The deskewed wb_table_image.
    tuple[MatLike, ...]
        Other images rotated by the same angle.
    """

    # Detect vertical lines and calculate the average angle

    lines_median_angle = nivo_lines_angle(
        wb_table_image,
        orientation="vertical",
        configuration=configuration,
    )
    if not lines_median_angle:
        return None, wb_table_image, *oth_images

    # Rotate the image to deskew
    (h, w) = wb_table_image.shape[:2]
    center = (h // 2, w // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, 90 - lines_median_angle, 1.0)
    wb_rotated = cv2.warpAffine(
        wb_table_image,
        rotation_matrix,
        dsize=[w, h],
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=[0],
    )

    oth_rotated: list[MatLike] = []
    for image in oth_images:
        (h, w) = image.shape[:2]
        center = (h // 2, w // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, 90 - lines_median_angle, 1.0)
        oth_rotated.append(
            cv2.warpAffine(
                image,
                rotation_matrix,
                (w, h),
                flags=cv2.INTER_CUBIC,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=[0],
            )
        )

    return lines_median_angle, wb_rotated, *oth_rotated


def my_table_struct(bw_image: MatLike) -> MatLike:
    """
    Extract table structure using morphological operations.

    Parameters
    ----------
    bw_image : MatLike
        Binary image where foreground is white.

    Returns
    -------
    MatLike
        Image with detected table structure.
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
    image_in_grayscale: MatLike, configuration: BinarizationConfiguration
) -> MatLike:
    """
    Apply adaptive thresholding to create a binary image.

    Parameters
    ----------
    image_in_grayscale : MatLike
        Grayscale image.
    configuration : BinarizationConfiguration
        Binarization configuration.

    Returns
    -------
    MatLike
        Binary image.
    """
    binarized_image = cv2.adaptiveThreshold(
        image_in_grayscale,
        255,
        configuration.adaptive_threshold_type,
        cv2.THRESH_BINARY,
        configuration.region_side,
        configuration.threshold_c,
    )
    return binarized_image


def ms_threshold(image: MatLike, configuration: ThresholdConfiguration) -> MatLike:
    """
    Apply Otsu's thresholding and morphological closing.

    Parameters
    ----------
    image : MatLike
        Input image (grayscale or color).
    configuration : ThresholdConfiguration
        Threshold configuration.

    Returns
    -------
    MatLike
        Thresholded image.
    """
    if image.ndim > 2:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    kernel = np.ones(configuration.kernel_shape, np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    return thresh


def preprocess(
    image: MatLike,
    deskew_method: Literal["nivo", "img2table"],
    configuration: PreprocessConfiguration,
) -> tuple[MatLike, MatLike, MatLike, MatLike]:
    """
    Preprocess image: rotate, convert to grayscale, binarize, threshold.

    Parameters
    ----------
    image : MatLike
        Input image.
    configuration : PreprocessConfiguration
        Preprocess configuration.
    deskew_method : Literal["nivo", "img2table"]
        Deskew method to use.

    Returns
    -------
    tuple[MatLike, MatLike, MatLike, MatLike]
        tuple of (rotated, rotated_threshold, rotated_binarized, rotated_table_struct) images.
    """
    # Fix rotation. By default and as a starting point use the img2table method
    image = fix_rotation_image(image)[0]
    # If using the custom nivo method, deskew the image and compare results. `image` is the rotated image
    # TODO: this is probably still improvable
    if deskew_method == "nivo":
        image_in_grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        thresh = ms_threshold(image_in_grayscale, configuration.threshold_configuration)
        # Check if deskewing improved the result, i.e. if the angle is closer to 90 degrees after deskewing
        angle_before = nivo_lines_angle(
            thresh, "vertical", configuration.angle_detection_configuration
        )
        _, thresh, nivo_image = deskew_nivo(
            thresh, image, configuration=configuration.angle_detection_configuration
        )
        angle_after = nivo_lines_angle(
            thresh, "vertical", configuration.angle_detection_configuration
        )
        if angle_after is not None and (
            angle_before is None or abs(90 - angle_before) > abs(90 - angle_after)
        ):
            image = nivo_image
    elif deskew_method != "img2table":
        raise ValueError(f"Invalid deskew method: {deskew_method}")  # pyright: ignore[reportUnreachable]

    image_in_grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply adaptive thresholding
    binarized_image = ms_binary(
        image_in_grayscale, configuration.binarization_configuration
    )

    # Apply Otsu's thresholding
    thresh = ms_threshold(image_in_grayscale, configuration.threshold_configuration)

    return (
        image,
        thresh,
        binarized_image,
        my_table_struct(thresh),
    )


def extract_contours_boxes(image: MatLike) -> list[Rect]:
    """
    Extract bounding boxes from contours in the image.

    Parameters
    ----------
    image : MatLike
        Processed image.

    Returns
    -------
    list[Rect]
        List of bounding rectangles.
    """
    contours = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]
    return [
        cv2.boundingRect(contour)
        for contour in contours
        if cv2.contourArea(contour) > 4
    ]
