"""Parts of the code were inspired by MeteoSaver (https://github.com/VUB-HYDR/MeteoSaver). Credit goes to the authors."""

from itertools import pairwise
from typing import Sequence

import numpy as np
import polars as pl
import polars_distance as pld
from cv2.typing import MatLike, Rect
from numpy.typing import NDArray

from .configuration.table_and_cell_detection import (
    WordBlobsCreationParameters,
)
from .image_processing import extract_contours_boxes
from .roi_utilities import easyrect2rect, extract, rect2easy
from .table_detection import create_word_blobs


def filter_by_size(
    input_boxes: Sequence[Rect], char_shape: tuple[int, int]
) -> list[Rect]:
    """Filter boxes by minimum size.

    Args:
        input_boxes: list of rectangles
        char_shape: (width, height) minimum size

    Returns:
        Filtered boxes
    """
    return list(
        filter(
            lambda box: box[2] >= char_shape[0] and box[3] >= char_shape[1], input_boxes
        )
    )


def merge_boxes(boxes: list[Rect]) -> Rect | None:
    """Merge multiple boxes into bounding box.

    Args:
        boxes: list of rectangles

    Returns:
        Merged rectangle or None if empty
    """
    if len(boxes) == 0:
        return None
    ls = [box[0] for box in boxes]
    us = [box[1] for box in boxes]
    rs = [box[0] + box[2] for box in boxes]
    ds = [box[1] + box[3] for box in boxes]
    bounds = [min(ls), min(us), max(rs), max(ds)]
    return [bounds[0], bounds[1], bounds[2] - bounds[0], bounds[3] - bounds[1]]


def merge_and_filter_boxes(
    input_boxes: Sequence[Rect], rows_positions: Sequence[int], char_height: int
) -> list[Rect]:
    """Merge and filter boxes aligned with rows.

    Args:
        input_boxes: list of rectangles
        rows_positions: Row center y-coordinates
        char_height: Character height

    Returns:
        Merged and filtered boxes
    """
    input_boxes = sorted(input_boxes, key=lambda b: b[1])
    rows_positions = sorted(rows_positions)
    rows_heights = [rows_positions[0]] + list(
        map(lambda p: p[1] - p[0], pairwise(rows_positions))
    )
    row_height = min(min(rows_heights), 3 * char_height)
    output_boxes: list[Rect] = []
    for row_position in rows_positions:
        matching_boxes = list(
            filter(
                lambda box: box[1][1]
                <= row_position - row_height
                <= box[1][1] + box[1][3]
                or box[1][1] <= row_position <= box[1][1] + box[1][3],
                enumerate(input_boxes),
            )
        )
        indices = [i for i, _ in matching_boxes]
        matching_boxes = [b for _, b in matching_boxes]
        matching_box = merge_boxes(matching_boxes)
        if matching_box:
            output_boxes.append(matching_box)
        input_boxes = input_boxes[max(indices) + 1 :]
    return output_boxes


def detect_station_boxes(
    column_image: MatLike, char_shape: tuple[int, int], rows_centers: list[int]
) -> list[Rect]:
    """Detect station name boxes in first column.

    Args:
        column_image: First column image
        char_shape: Character dimensions
        rows_centers: Row center positions

    Returns:
        list of station box rectangles
    """
    word_blobs = create_word_blobs(
        column_image,
        WordBlobsCreationParameters(gap_kernel_shape=(char_shape[0] // 2, 1)),
    )
    word_boxes = list(extract_contours_boxes(word_blobs))
    word_boxes.sort(key=lambda r: r[1])
    filtered_wboxes = filter_by_size(word_boxes, char_shape)
    filtered_wboxes = merge_and_filter_boxes(
        filtered_wboxes, rows_centers, char_shape[1]
    )
    return filtered_wboxes


def associate_closest_station_names(
    results: Sequence[str], anagrafica: Sequence[str]
) -> list[dict]:
    """Match OCR results to known station names using Levenshtein distance.

    Args:
        results: OCR recognized names
        anagrafica: Known station names

    Returns:
        list of dicts with matched names and similarity scores
    """
    anagrafica_df = pl.DataFrame({"name_anagrafica": anagrafica}).with_columns(
        simplified_name=pl.col("name_anagrafica")
        .str.to_lowercase()
        .str.strip_chars()
        .str.replace_all(r"\s+", " ")
    )
    results_df = (
        (
            pl.DataFrame({"ocr_name": results})
            .with_columns(pl.row_index())
            .with_columns(
                simplified_name=pl.col("ocr_name")
                .str.to_lowercase()
                .str.strip_chars()
                .str.replace_all(r"\s+", " ")
            )
            .join(anagrafica_df, how="cross", suffix="_anagrafica")
        )
        .with_columns(
            distance=(
                pl.col("simplified_name")
                .dist_str.levenshtein("simplified_name_anagrafica")  # type: ignore
                .cast(pl.Int32)
            )
        )
        .sort("index", "distance")
        .group_by("index", maintain_order=True)
        .head(1)
        .sort("index")
        .with_columns(similarity=-pl.col("distance"))
    )
    return (
        results_df.sort("index")
        .select(name="name_anagrafica", string_similarity="similarity")
        .to_dicts()
    )


def compute_name_rows(boxes: list[Rect]) -> list[int]:
    """Sort box indices by y-coordinate.

    Args:
        boxes: list of rectangles

    Returns:
        Sorted indices
    """
    ys = [b[1] for b in boxes]
    return np.argsort(ys).tolist()


def transpose_recognition_results(recognition: list) -> dict:
    """Transpose recognition results to group by key.

    Args:
        recognition: list of result dicts

    Returns:
        Dict mapping keys to lists
    """
    return {key: [r[key] for r in recognition] for key in recognition[0].keys()}


def merge_easypolys(polys: list | NDArray) -> list[int]:
    """Merge multiple easyocr polygons into single rectangle.

    Args:
        polys: list of polygons

    Returns:
        Merged rectangle [x1, x2, y1, y2]
    """
    polys = np.array(polys)
    return [
        int(polys[..., 0].min()),
        int(polys[..., 0].max()),
        int(polys[..., 1].min()),
        int(polys[..., 1].max()),
    ]


def process_easyocr_readtext_result(cell_results: list[dict]) -> dict:
    """Process easyocr readtext results for a single cell.

    Args:
        cell_results: list of recognition results

    Returns:
        Processed result with boxes, text, confidence
    """
    return {
        "boxes": easyrect2rect(
            merge_easypolys([result["boxes"] for result in cell_results])
        ),
        "text": " ".join([r["text"] for r in cell_results]),
        "confident": float(np.mean([r["confident"] for r in cell_results])),
    }


def process_easyocr_recognize_result(box_results: list[dict]) -> list[dict]:
    """Process easyocr recognize results.

    Args:
        box_results: list of box results

    Returns:
        Processed results
    """
    return [process_easyocr_readtext_result([box_result]) for box_result in box_results]
