"""ROI (Region of Interest) utilities for NIVO table cell extraction. Parts of the code were inspired by MeteoSaver (https://github.com/VUB-HYDR/MeteoSaver). Credit goes to the authors."""

from itertools import pairwise
from math import ceil, floor
from typing import Any

import cv2
import numpy as np
from cv2.typing import MatLike, Rect
from numba import njit
from numpy.typing import NDArray
from sklearn.cluster import KMeans

from .configuration.table_and_cell_detection import WordBlobsCreationParameters
from .table_detection import create_word_blobs


def generate_roi(
    row_position: int,
    column_separators: tuple[int, int],
    char_height: int,
    extra_width: int,
) -> Rect:
    """Generate ROI for a single cell.

    Args:
        row_position: Center y-coordinate of row
        column_separators: (x1, x2) column boundaries
        char_height: Character height
        extra_width: Extra width padding

    Returns:
        Rectangle (x, y, width, height)
    """
    row_height = ceil(int(1.5 * char_height))
    column_width = (column_separators[1] - column_separators[0]) + extra_width
    return [
        column_separators[0] - extra_width // 2,
        row_position - int(floor(row_height / 2)),
        column_width,
        row_height,
    ]


def generate_roi_grid(
    row_positions: list[int],
    column_separators: list[int],
    char_height: int,
    extra_width: int,
) -> list[list[Rect]]:
    """Generate grid of ROIs for all cells.

    Args:
        row_positions: list of row center y-coordinates
        column_separators: list of column x-coordinates
        char_height: Character height
        extra_width: Extra width padding

    Returns:
        2D list of rectangles [columns][rows]
    """
    row_positions = sorted(row_positions)
    column_separators = sorted(column_separators)
    return [
        [
            generate_roi(row_position, column_sep, char_height, extra_width)
            for row_position in row_positions
        ]
        for column_sep in pairwise(column_separators)
    ]


def roi_grid_coordinates(
    rois: list[Rect], n_rows: int, n_cols: int
) -> list[tuple[int, int]]:
    """Label ROIs by their grid position using KMeans clustering.

    Args:
        rois: list of rectangles
        n_rows: Number of rows in grid
        n_cols: Number of columns in grid

    Returns:
        list of (row, col) tuples for each ROI
    """
    if not rois:
        return []

    # Extract center points of all ROIs
    centers = np.array(
        [[rect[0] + rect[2] / 2, rect[1] + rect[3] / 2] for rect in rois]
    )

    # Cluster by Y-coordinate to find rows
    y_coords = centers[:, 1].reshape(-1, 1)
    kmeans_rows = KMeans(n_clusters=n_rows, n_init=10, random_state=42)
    row_labels = kmeans_rows.fit_predict(y_coords)

    # Sort row clusters by their center Y-coordinate
    row_centers = kmeans_rows.cluster_centers_.flatten()
    row_order = np.argsort(row_centers)
    row_mapping = {
        old_label: new_label for new_label, old_label in enumerate(row_order)
    }
    row_labels = np.array([row_mapping[label] for label in row_labels])

    # Cluster by X-coordinate to find columns
    x_coords = centers[:, 0].reshape(-1, 1)
    kmeans_cols = KMeans(n_clusters=n_cols, n_init=10, random_state=42)
    col_labels = kmeans_cols.fit_predict(x_coords)

    # Sort column clusters by their center X-coordinate
    col_centers = kmeans_cols.cluster_centers_.flatten()
    col_order = np.argsort(col_centers)
    col_mapping = {
        old_label: new_label for new_label, old_label in enumerate(col_order)
    }
    col_labels = np.array([col_mapping[label] for label in col_labels])

    # Combine results
    return [(int(row_labels[i]), int(col_labels[i])) for i in range(len(rois))]


def is_rect_contained(inner: Rect, outer: Rect) -> bool:
    """Check if inner rectangle is contained in outer rectangle.

    Args:
        inner: Inner rectangle
        outer: Outer rectangle

    Returns:
        True if inner is strictly contained in outer
    """
    inner_x1, inner_y1 = inner[0], inner[1]
    inner_x2, inner_y2 = inner[0] + inner[2], inner[1] + inner[3]

    outer_x1, outer_y1 = outer[0], outer[1]
    outer_x2, outer_y2 = outer[0] + outer[2], outer[1] + outer[3]

    return (
        inner_x1 >= outer_x1
        and inner_y1 >= outer_y1
        and inner_x2 <= outer_x2
        and inner_y2 <= outer_y2
    )


def label_contained_roi(
    container_rois: list[Rect], labels: list[Any], child_roi: Rect
) -> Any:
    """Find label of container that contains child ROI.

    Args:
        container_rois: list of container rectangles
        labels: list of labels corresponding to containers
        child_roi: ROI to label

    Returns:
        Label of containing container
    """
    for container_roi, label in zip(container_rois, labels):
        if is_rect_contained(child_roi, container_roi):
            return label


def autocrop(image: MatLike) -> MatLike:
    """Remove padding from image.

    Args:
        image: Input image

    Returns:
        Cropped image
    """
    is_fg = image > 0
    cols_with_content = np.argwhere(is_fg.any(axis=0)).flatten()
    x_from, x_to = cols_with_content.min(), cols_with_content.max()
    rows_with_content = np.argwhere(is_fg.any(axis=1)).flatten()
    y_from, y_to = rows_with_content.min(), rows_with_content.max()
    return image[y_from : y_to + 1, x_from : x_to + 1]


def autocrop_axis(a: NDArray, axis: int):
    a_with_content = np.argwhere(a.any(axis)).flatten()
    if len(a_with_content) > 0:
        return int(a_with_content.min()), int(a_with_content.max())
    return 0, len(a_with_content)


def autocrop_roi(roi: Rect, image: MatLike) -> Rect:
    """Autocrop a region of interest within an image.

    Args:
        roi: Region of interest (x, y, width, height)
        image: Full image

    Returns:
        Cropped ROI coordinates
    """
    image = extract(image, roi)
    is_fg = image > 0
    x_from, x_to = autocrop_axis(is_fg, axis=0)
    y_from, y_to = autocrop_axis(is_fg, axis=1)
    return [roi[0] + x_from, roi[1] + y_from, x_to - x_from, y_to - y_from]


def pad_roi(roi: Rect, padding: int | tuple[int, int]) -> Rect:
    """Add padding to region of interest.

    Args:
        roi: Region (x, y, width, height)
        padding: Padding size in pixels
        roi_area: (height, width) of full image

    Returns:
        Padded ROI
    """
    if isinstance(padding, int):
        pad_x, pad_y = padding, padding
    else:
        pad_x, pad_y = padding
    x, y, w, h = roi

    padded_x1 = max(x - pad_x, 0)
    padded_y1 = max(y - pad_y, 0)
    return [
        padded_x1,
        padded_y1,
        w + 2 * pad_x,
        h + 2 * pad_y,
    ]


def rect2easy(rect: Rect) -> list[int]:
    """Convert OpenCV rect to easyocr format [x1, x2, y1, y2].

    Args:
        rect: OpenCV rectangle (x, y, w, h)

    Returns:
        Easyocr rectangle format
    """
    x, y, w, h = rect
    return [x, x + w, y, y + h]


def easyrect2rect(eo_rect: list[int]) -> Rect:
    """Convert easyocr format to OpenCV rect.

    Args:
        eo_rect: Easyocr rectangle [x1, x2, y1, y2]

    Returns:
        OpenCV rectangle (x, y, w, h)
    """
    x1, x2, y1, y2 = eo_rect
    return [x1, y1, x2 - x1, y2 - y1]


def resize_roi_to_largest_connected_region(
    roi: Rect,
    binarized_image: MatLike,
    word_blobs_parameters: WordBlobsCreationParameters,
) -> Rect | None:
    """Resize ROI to the largest connected region of foreground pixels.

    This function takes a ROI and a binarized image (white foreground, black background),
    applies the word blobs technique to connect nearby foreground regions, finds the
    largest connected component within the ROI, and returns a new ROI that tightly
    bounds that component.

    Args:
        roi: Region of interest (x, y, width, height)
        binarized_image: Binary image with white (255) foreground and black (0) background
        word_blobs_parameters: Parameters for word blob creation

    Returns:
        New ROI bounding the largest connected region, or None if no foreground found
    """
    # Extract the ROI region from the image
    roi_image = extract(binarized_image, roi)

    if roi_image.shape[0] == 0 or roi_image.shape[1] == 0:
        return roi

    # Apply word blobs technique to connect nearby foreground regions
    roi_with_blobs = create_word_blobs(roi_image, word_blobs_parameters)

    # Find connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        roi_with_blobs, connectivity=8
    )

    # If no components found (only background), return None
    if num_labels <= 1:
        return None

    # Find the largest component (excluding background which is label 0)
    # stats columns: [x, y, width, height, area]
    largest_component_idx = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])

    # Get bounding box of largest component (relative to ROI)
    x_rel = stats[largest_component_idx, cv2.CC_STAT_LEFT]
    y_rel = stats[largest_component_idx, cv2.CC_STAT_TOP]
    w_rel = stats[largest_component_idx, cv2.CC_STAT_WIDTH]
    h_rel = stats[largest_component_idx, cv2.CC_STAT_HEIGHT]

    # Convert to absolute coordinates
    x_abs = roi[0] + x_rel
    y_abs = roi[1] + y_rel

    return [int(x_abs), int(y_abs), int(w_rel), int(h_rel)]


def expand_roi_atleast(roi: Rect, atleast: tuple[int, int]):
    _, _, w, h = roi
    exp_w, exp_h = atleast
    pad_x = int(ceil(max((exp_w - w) / 2, 0)))
    pad_y = int(ceil(max((exp_h - h) / 2, 0)))
    return pad_roi(roi, (pad_x, pad_y))


def extract(image: MatLike, rect: Rect) -> MatLike:
    """Extract rectangular region from image.

    Args:
        image: Input image
        rect: Rectangle (x, y, width, height)

    Returns:
        Extracted region
    """
    x, y, w, h = rect
    return image[
        max(y, 0) : min(y + h, image.shape[0]),
        max(x, 0) : min(x + w, image.shape[1]),
    ]
    # return image[y : y + h, x : x + w]
