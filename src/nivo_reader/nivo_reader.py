"""Main NIVO table reader module."""

from typing import Any, Callable
from pathlib import Path

import easyocr
import numpy as np
from cv2.typing import MatLike, Rect
from itertools import batched

from . import ocr_processing
from .configuration.preprocessing import (
    PreprocessingParameters,
    ThresholdParameters,
)
from .configuration.table_and_cell_detection import (
    LinesExtractionParameters,
    WordBlobsCreationParameters,
)
from .excel_output import (
    draw_bounding_boxes,
    draw_straight_lines,
    save_artifacts,
    write_tables_to_excel,
)
from .image_processing import preproc
from .ocr_processing import (
    associate_closest_station_names,
    compute_name_rows,
    detect_station_boxes,
    easyocr_names_reader,
)
from .roi_utilities import (
    autocrop_roi,
    expand_roi_atleast,
    generate_roi_grid,
    pad_roi,
    resize_roi_to_largest_connected_region,
)
from .table_detection import (
    cut_out_tables,
    detect_column_separators,
    detect_rows_positions,
    remove_lines_from_image,
    try_detect_table_rect,
)

value_readers = {
    reader: getattr(ocr_processing, f"{reader}_values_reader")
    for reader in ["tesseract", "easyocr"]
}


def read_station_names(
    image: MatLike,
    rows_centers: list[int],
    column_separators: list[int],
    char_shape: tuple[int, int],
    reader: Callable[
        [MatLike, list[Rect]], tuple[list[str | None], list[float | None], list[Rect]]
    ],
    roi_padding: int,
    anagrafica: list[str],
) -> tuple[list[str | None], list[float | None], list[Rect]]:
    """Read station names from first column.

    Args:
        image: Table image
        rows_centers: Row center positions
        column_separators: Column x-coordinates
        char_shape: Character (width, height)
        reader: OCR reader callable
        roi_padding: Padding around ROIs
        anagrafica: Known station names

    Returns:
        tuple of (names, similarities, boxes)
    """
    first_column = image[:, column_separators[0] : column_separators[1]]
    names_wboxes = list(
        map(
            lambda box: pad_roi(autocrop_roi(box, image), roi_padding),
            detect_station_boxes(first_column, char_shape, rows_centers),
        )
    )

    names_wboxes = sorted(names_wboxes, key=lambda b: b[1])
    ocr_names, names_confidences, ocr_name_boxes = reader(image, names_wboxes)

    names_rows = compute_name_rows(ocr_name_boxes)
    ocr_names: list[str | None] = np.array(ocr_names)[names_rows].tolist()
    names_confidences: list[float | None] = np.array(names_confidences)[
        names_rows
    ].tolist()
    ocr_name_boxes: list[Rect] = np.array(ocr_name_boxes)[names_rows].tolist()

    anagrafica_closest = associate_closest_station_names(ocr_names, anagrafica)
    anagrafica_names: list[str | None] = list(
        map(lambda d: d["name"], anagrafica_closest)
    )
    anagrafica_similarities: list[float | None] = list(
        map(lambda d: float(d["string_similarity"]), anagrafica_closest)
    )

    return anagrafica_names, anagrafica_similarities, ocr_name_boxes


def prepare_value_roi(
    roi: Rect,
    image: MatLike,
    character_shape: tuple[int, int],
    parameters: WordBlobsCreationParameters,
    padding: int,
):
    largest_region = resize_roi_to_largest_connected_region(roi, image, parameters)
    if largest_region is None:
        largest_region = roi
    largest_region = autocrop_roi(largest_region, image)
    largest_region = expand_roi_atleast(largest_region, character_shape)

    return pad_roi(largest_region, padding)


def populate_with_reading_results(
    results: tuple[list[str | None], list[float | None], list[Rect]],
    reader: Callable[
        [MatLike, list[Rect]], tuple[list[str | None], list[float | None], list[Rect]]
    ],
    image: MatLike,
    rois: list[Rect],
    threshold: float,
) -> None:
    """Fill results with OCR readings, only when confidence is above threshold.

    Filters out ROIs that already have satisfying results to avoid unnecessary
    OCR processing. Results are placed in their proper positions.

    Args:
        results: Tuple of (readings, confidences, boxes) lists to populate
        reader: OCR reader callable that takes (image, rois) and returns
            (readings, confidences, boxes)
        image: Image to process
        rois: List of ROI rectangles
        threshold: Confidence threshold - only results above this are kept
    """
    # Filter ROIs that don't already have satisfying results
    # Keep track of original indices to map results back correctly
    filtered_rois: list[Rect] = []
    original_indices: list[int] = []

    for i, roi in enumerate(rois):
        if results[0][i] is None:  # Only process ROIs that don't have results yet
            filtered_rois.append(roi)
            original_indices.append(i)

    # If all ROIs already have results, skip OCR processing
    if not filtered_rois:
        return

    # Call reader with only the filtered ROIs
    readings, confidences, boxes = reader(image, filtered_rois)

    # Map results back to their original positions
    for filtered_idx, original_idx in enumerate(original_indices):
        reading = readings[filtered_idx]
        confidence = confidences[filtered_idx]
        box = boxes[filtered_idx]

        # Only fill in results if confidence is above threshold
        if confidence is not None and confidence > threshold:
            if results[0][original_idx] is None:  # Double-check it's still None
                results[0][original_idx] = reading
                results[1][original_idx] = confidence
                results[2][original_idx] = box

def _grouped(values: list[Any], n: int) -> list[list[Any]]:
    return list(map(list, batched(values, n)))

def read_values(
    image: MatLike,
    rows_centers: list[int],
    column_separators: list[int],
    number_char_shape: tuple[int, int],
    readers: list[
        Callable[
            [MatLike, list[Rect]],
            tuple[list[str | None], list[float | None], list[Rect]],
        ]
    ],
    confidence_thresholds: list[float],
    roi_padding: int,
    extra_width: int,
) -> tuple[list[list[str | None]], list[list[float | None]], list[list[Rect]]]:
    """Read values from table cells.

    Args:
        image: Table image
        rows_centers: Row center positions
        column_separators: Column x-coordinates
        number_char_shape: Character (width, height)
        readers: List of OCR reader callables
        confidence_thresholds: List of confidence thresholds
        roi_padding: Padding around ROIs
        extra_width: Extra width padding

    Returns:
        tuple of (values, confidences, boxes) as 2D lists [col][row]
    """
    rois_grid = generate_roi_grid(
        rows_centers, column_separators[1:], number_char_shape[1], extra_width
    )
    n_rows = len(rows_centers)

    # Flatten for processing
    rois_flat = [roi for col in rois_grid for roi in col]

    rois_flat = list(
        map(
            lambda roi: prepare_value_roi(
                roi,
                image,
                number_char_shape,
                WordBlobsCreationParameters(gap_iterations=2, simple_iterations=0),
                roi_padding,
            ),
            rois_flat,
        )
    )

    results_flat: tuple[list[str | None], list[float | None], list[Rect]] = (
        [None] * len(rois_flat),
        [None] * len(rois_flat),
        rois_flat,
    )

    while readers:
        reader = readers.pop(0)
        threshold = confidence_thresholds.pop(0)
        populate_with_reading_results(results_flat, reader, image, rois_flat, threshold)

    return _grouped(results_flat[0], n_rows), _grouped(results_flat[1], n_rows), _grouped(results_flat[2], n_rows)


def read_nivo_table(
    original_image: MatLike,
    excel_out_path,
    clips: tuple[int, int, int, int],
    table_shape: tuple[int, int],
    anagrafica: list[str],
    easyreader: easyocr.Reader,
    station_char_shape: tuple[int, int] = (12, 10),
    number_char_shape: tuple[int, int] = (12, 20),
    roi_padding: int = 3,
    nchars_threshold: int = 30,
    extra_width: int = 6,
    low_confidence_threshold: float = 0.7,
    debug_dir: Path | None = None,
) -> None:
    """Extract NIVO table data from image and write to Excel.

    Args:
        original_image: Input image
        excel_out_path: Output Excel file path
        ocr: Initialized EasyOCR reader
        clips: (up, down, left, right) clipping margins
        table_shape: (width, height) of table
        anagrafica: list of known station names
        station_char_shape: Character (width, height) for station names
        number_char_shape: Character (width, height) for numbers
        roi_padding: Padding around ROIs in pixels
        nchars_threshold: Minimum character count for peak detection
        extra_width: Extra width for cell ROIs
        low_confidence_threshold: Confidence threshold for marking cells
        debug_dir: Directory for debug artifacts (optional)

    Raises:
        Exception: If table cannot be detected
    """
    if debug_dir:
        debug_dir = Path(debug_dir)
        debug_dir.mkdir(parents=True, exist_ok=True)
        debug_image = original_image.copy()

    # Preprocessing
    image, threshold_image, binarized_image, _ = preproc(
        original_image,
        PreprocessingParameters(),
        deskew_method="nivo",
    )

    # Detect table rectangle
    rect = try_detect_table_rect(image, table_shape, ThresholdParameters())

    if rect is None:
        raise ValueError("Could not detect table rectangle in image")

    # Cut out table
    _, binarized_subtable = cut_out_tables(binarized_image, rect, clips)
    _, threshold_subtable = cut_out_tables(threshold_image, rect, clips)

    # Remove lines
    binarized_subtable_wo_lines = remove_lines_from_image(
        255 - binarized_subtable,
        LinesExtractionParameters(),
    )

    # Detect rows and columns
    rows_centers = detect_rows_positions(
        binarized_subtable_wo_lines, nchars_threshold, number_char_shape
    ).tolist()
    cols_separators = detect_column_separators(threshold_subtable, number_char_shape[0])

    # Save debug artifacts if requested
    if debug_dir:
        save_artifacts(
            {
                "01_table_rect": draw_bounding_boxes(debug_image, [rect]), # pyright: ignore[reportPossiblyUnboundVariable]
                "02_binarized_subtable": binarized_subtable,
                "03_binarized_subtable_wo_lines": binarized_subtable_wo_lines,
                "04_lines": draw_straight_lines(
                    draw_straight_lines(
                        binarized_subtable_wo_lines, rows_centers, "horizontal"
                    ),
                    cols_separators,
                    "vertical",
                ),
            },
            debug_dir,
        )

    # Read station names
    ocr_names, names_anagrafica_similarities, ocr_name_boxes = read_station_names(
        binarized_subtable_wo_lines,
        rows_centers,
        cols_separators,
        station_char_shape,
        easyocr_names_reader(easyreader),
        roi_padding,
        anagrafica,
    )

    if debug_dir:
        save_artifacts(
            {
                "05_station_names_bboxes": draw_bounding_boxes(
                    binarized_subtable_wo_lines, ocr_name_boxes
                )
            },
            debug_dir,
        )

    # Read values
    ocr_values_2d, values_confidences_2d, ocr_value_boxes_2d = read_values(
        binarized_subtable_wo_lines,
        rows_centers,
        cols_separators,
        number_char_shape,
        [value_readers["easyocr"](easyreader)],
        [0.0],
        roi_padding,
        extra_width,
    )

    # Flatten results for Excel output and debug
    ocr_values = [val for col in ocr_values_2d for val in col]
    values_confidences = [conf for col in values_confidences_2d for conf in col]
    ocr_value_boxes = [box for col in ocr_value_boxes_2d for box in col]

    if debug_dir:
        save_artifacts(
            {
                "06_values_bboxes": draw_straight_lines(
                    draw_bounding_boxes(
                        binarized_subtable_wo_lines,
                        ocr_value_boxes,
                        thickness=1,
                    ),
                    rows_centers,
                    "horizontal",
                )
            },
            debug_dir,
        )

    # Assign grid coordinates
    values_coordinates: list[tuple[int, int]] = []
    for c_idx, col in enumerate(ocr_values_2d):
        for r_idx, _ in enumerate(col):
            values_coordinates.append((r_idx, c_idx))

    # Write to Excel
    write_tables_to_excel(
        ocr_values,
        values_confidences,
        values_coordinates,
        str(excel_out_path),
        ocr_names,
        names_anagrafica_similarities,
        low_confidence_threshold,
    )
