"""ROI (Region of Interest) utilities for NIVO table cell extraction."""

from typing import Any
from math import ceil, floor
from itertools import pairwise
import numpy as np
from cv2.typing import MatLike, Rect
from sklearn.cluster import KMeans

from .table_detection import extract
from .ocr_processing import autocrop


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


def read_roi(roi: Rect, image: MatLike, readers: list) -> list[str]:
    """Read text from ROI using multiple readers.

    Args:
        roi: Region of interest
        image: Image to read from
        readers: list of reader callables

    Returns:
        list of recognized texts
    """
    cell_content = np.pad(
        autocrop(extract(image, roi)), pad_width=3, mode="constant", constant_values=0
    )
    return [reader(cell_content) for reader in readers]
