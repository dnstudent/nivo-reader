"""ROI (Region of Interest) utilities for NIVO table cell extraction. Parts of the code were inspired by MeteoSaver (https://github.com/VUB-HYDR/MeteoSaver). Credit goes to the authors."""

from itertools import pairwise
from math import ceil, floor
from typing import Any

import cv2
import numpy as np
from cv2.typing import MatLike, Rect
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
    """
    Generate ROI for a single cell.

    Parameters
    ----------
    row_position : int
        Center y-coordinate of row.
    column_separators : tuple[int, int]
        (x1, x2) column boundaries.
    char_height : int
        Character height.
    extra_width : int
        Extra width padding.

    Returns
    -------
    Rect
        Rectangle (x, y, width, height).
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
    """
    Generate grid of ROIs for all cells.

    Parameters
    ----------
    row_positions : list[int]
        List of row center y-coordinates.
    column_separators : list[int]
        List of column x-coordinates.
    char_height : int
        Character height.
    extra_width : int
        Extra width padding.

    Returns
    -------
    list[list[Rect]]
        2D list of rectangles [columns][rows].
    """
    row_positions = sorted(row_positions)
    column_separators = sorted(column_separators)
    return [
        [
            generate_roi(row_position, column_seps, char_height, extra_width)
            for row_position in row_positions
        ]
        for column_seps in pairwise(column_separators)
    ]


def roi_grid_coordinates(
    rois: list[Rect], n_rows: int, n_cols: int
) -> list[tuple[int, int]]:
    """
    Label ROIs by their grid position using KMeans clustering.

    Parameters
    ----------
    rois : list[Rect]
        List of rectangles.
    n_rows : int
        Number of rows in grid.
    n_cols : int
        Number of columns in grid.

    Returns
    -------
    list[tuple[int, int]]
        List of (row, col) tuples for each ROI.
    """
    if not rois:
        return []

    # Extract center points of all ROIs
    centers = np.array(
        [[rect[0] + rect[2] / 2, rect[1] + rect[3] / 2] for rect in rois]
    )

    # Cluster by Y-coordinate to find rows
    y_coords = centers[:, 1].reshape(-1, 1)
    kmeans_rows = KMeans(n_clusters=n_rows, n_init=10)  # pyright: ignore[reportArgumentType]
    row_labels = kmeans_rows.fit_predict(y_coords)  # pyright: ignore[reportUnknownMemberType]

    # Sort row clusters by their center Y-coordinate
    row_centers = kmeans_rows.cluster_centers_.flatten()  # pyright: ignore[reportUnknownMemberType]
    row_order = np.argsort(row_centers)
    row_mapping = {
        old_label: new_label for new_label, old_label in enumerate(row_order)
    }
    row_labels = np.array([row_mapping[label] for label in row_labels])

    # Cluster by X-coordinate to find columns
    x_coords = centers[:, 0].reshape(-1, 1)
    kmeans_cols = KMeans(n_clusters=n_cols, n_init=10)  # pyright: ignore[reportArgumentType]
    col_labels = kmeans_cols.fit_predict(x_coords)  # pyright: ignore[reportUnknownMemberType]

    # Sort column clusters by their center X-coordinate
    col_centers = kmeans_cols.cluster_centers_.flatten()  # pyright: ignore[reportUnknownMemberType]
    col_order = np.argsort(col_centers)
    col_mapping = {
        old_label: new_label for new_label, old_label in enumerate(col_order)
    }
    col_labels = np.array([col_mapping[label] for label in col_labels])

    # Combine results
    return [(int(row_labels[i]), int(col_labels[i])) for i in range(len(rois))]


def is_rect_contained(inner: Rect, outer: Rect) -> bool:
    """
    Check if inner rectangle is contained in outer rectangle.

    Parameters
    ----------
    inner : Rect
        Inner rectangle.
    outer : Rect
        Outer rectangle.

    Returns
    -------
    bool
        True if inner is strictly contained in outer.
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
    """
    Find label of container that contains child ROI.

    Parameters
    ----------
    container_rois : list[Rect]
        List of container rectangles.
    labels : list[Any]
        List of labels corresponding to containers.
    child_roi : Rect
        ROI to label.

    Returns
    -------
    Any
        Label of containing container, or None if not found.
    """
    for container_roi, label in zip(container_rois, labels):
        if is_rect_contained(child_roi, container_roi):
            return label


def autocrop(image: MatLike) -> MatLike:
    """
    Remove padding from image.

    Parameters
    ----------
    image : MatLike
        Input image.

    Returns
    -------
    MatLike
        Cropped image.
    """
    is_fg = image > 0
    cols_with_content = np.argwhere(is_fg.any(axis=0)).flatten()
    x_from, x_to = cols_with_content.min(), cols_with_content.max()
    rows_with_content = np.argwhere(is_fg.any(axis=1)).flatten()
    y_from, y_to = rows_with_content.min(), rows_with_content.max()
    return image[y_from : y_to + 1, x_from : x_to + 1]


def autocrop_axis(a: NDArray[Any], axis: int) -> tuple[int, int]:
    """
    Find the start and end indices of content along a specific axis.

    Parameters
    ----------
    a : NDArray[Any]
        Input array (image or mask).
    axis : int
        The axis along which to find content (0 for columns, 1 for rows).

    Returns
    -------
    tuple[int, int]
        Start and end indices.
    """
    a_with_content = np.argwhere(a.any(axis)).flatten()
    if len(a_with_content) > 0:
        return int(a_with_content.min()), int(a_with_content.max())
    return 0, a.shape[1 - axis]


def autocrop_roi(roi: Rect, image: MatLike) -> Rect:
    """
    Autocrop a region of interest within an image.

    Parameters
    ----------
    roi : Rect
        Region of interest (x, y, width, height).
    image : MatLike
        Full image.

    Returns
    -------
    Rect
        Cropped ROI coordinates.
    """
    image = extract(image, roi)
    is_fg = image > 0
    x_from, x_to = autocrop_axis(is_fg, axis=0)
    y_from, y_to = autocrop_axis(is_fg, axis=1)
    return [roi[0] + x_from, roi[1] + y_from, x_to - x_from, y_to - y_from]


def pad_roi(roi: Rect, padding: int | tuple[int, int]) -> Rect:
    """
    Add padding to region of interest.

    Parameters
    ----------
    roi : Rect
        Region (x, y, width, height).
    padding : int | tuple[int, int]
        Padding size in pixels. If int, same padding for width and height.

    Returns
    -------
    Rect
        Padded ROI.
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
    """
    Convert OpenCV rect to easyocr format [x1, x2, y1, y2].

    Parameters
    ----------
    rect : Rect
        OpenCV rectangle (x, y, w, h).

    Returns
    -------
    list[int]
        Easyocr rectangle format.
    """
    x, y, w, h = rect
    return [x, x + w, y, y + h]


def easyrect2rect(eo_rect: list[int]) -> Rect:
    """
    Convert easyocr format to OpenCV rect.

    Parameters
    ----------
    eo_rect : list[int]
        Easyocr rectangle [x1, x2, y1, y2].

    Returns
    -------
    Rect
        OpenCV rectangle (x, y, w, h).
    """
    x1, x2, y1, y2 = eo_rect
    return [x1, y1, x2 - x1, y2 - y1]


def resize_roi_to_largest_connected_region(
    roi: Rect,
    binarized_image: MatLike,
    word_blobs_parameters: WordBlobsCreationParameters,
) -> Rect | None:
    """
    Resize ROI to the largest connected region of foreground pixels.

    This function takes a ROI and a binarized image (white foreground, black background),
    applies the word blobs technique to connect nearby foreground regions, finds the
    largest connected component within the ROI, and returns a new ROI that tightly
    bounds that component.

    Parameters
    ----------
    roi : Rect
        Region of interest (x, y, width, height).
    binarized_image : MatLike
        Binary image with white (255) foreground and black (0) background.
    word_blobs_parameters : WordBlobsCreationParameters
        Parameters for word blob creation.

    Returns
    -------
    Rect | None
        New ROI bounding the largest connected region, or None if no foreground found.
    """
    # Extract the ROI region from the image
    roi_image = extract(binarized_image, roi)

    if roi_image.shape[0] == 0 or roi_image.shape[1] == 0:
        return roi

    # Apply word blobs technique to connect nearby foreground regions
    roi_with_blobs = create_word_blobs(roi_image, word_blobs_parameters)

    # Find connected components
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(
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


def expand_roi_atleast(roi: Rect, atleast: tuple[int, int]) -> Rect:
    """
    Expand ROI to be at least expected size.

    Parameters
    ----------
    roi : Rect
        Input rectangle (x, y, w, h).
    atleast : tuple[int, int]
        Minimum (width, height).

    Returns
    -------
    Rect
        Expanded rectangle.
    """
    _, _, w, h = roi
    exp_w, exp_h = atleast
    pad_x = int(ceil(max((exp_w - w) / 2, 0)))
    pad_y = int(ceil(max((exp_h - h) / 2, 0)))
    return pad_roi(roi, (pad_x, pad_y))


def extract(image: MatLike, rect: Rect) -> MatLike:
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
    x, y, w, h = rect
    return image[
        max(y, 0) : min(y + h, image.shape[0]),
        max(x, 0) : min(x + w, image.shape[1]),
    ]
    # return image[y : y + h, x : x + w]


def prepare_value_roi(
    roi: Rect,
    image: MatLike,
    character_shape: tuple[int, int],
    parameters: WordBlobsCreationParameters,
    padding: int,
):
    """
    Prepare ROI for value reading by resizing to largest connected region and padding.

    Parameters
    ----------
    roi : Rect
        Initial ROI.
    image : MatLike
        Image containing the ROI.
    character_shape : tuple[int, int]
        Expected character shape (width, height) to ensure minimum size.
    parameters : WordBlobsCreationParameters
        Parameters for word detection.
    padding : int
        Padding to add around the result.

    Returns
    -------
    Rect
        Prepared ROI.
    """
    largest_region = resize_roi_to_largest_connected_region(roi, image, parameters)
    if largest_region is None:
        largest_region = roi
    largest_region = autocrop_roi(largest_region, image)
    largest_region = expand_roi_atleast(largest_region, character_shape)

    return pad_roi(largest_region, padding)
