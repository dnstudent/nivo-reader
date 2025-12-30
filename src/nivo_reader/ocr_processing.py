"""Parts of the code were inspired by MeteoSaver (https://github.com/VUB-HYDR/MeteoSaver). Credit goes to the authors."""

from itertools import takewhile, pairwise, batched
from string import ascii_letters
import logging
from typing import Any, Callable

import easyocr
import numpy as np
import polars as pl
import polars_distance as pld  # noqa: F401  # pyright: ignore[reportUnusedImport]
import pytesseract
from cv2.typing import MatLike, Rect
from numpy.typing import NDArray

from .configuration.table_and_cell_detection import (
    WordBlobsCreationConfiguration,
)
from .image_processing import extract_contours_boxes
from .roi_utilities import (
    easyrect2rect,
    extract,
    rect2easy,
    pad_roi,
    autocrop_roi,
    generate_roi_grid,
    prepare_value_roi,
)
from .table_detection import create_word_blobs
from .excel_output import draw_bounding_boxes

logger = logging.getLogger(__name__)


def filter_by_size(input_boxes: list[Rect], char_shape: tuple[int, int]) -> list[Rect]:
    """
    Filter boxes by minimum size.

    Parameters
    ----------
    input_boxes : list[Rect]
        List of rectangles.
    char_shape : tuple[int, int]
        (width, height) minimum size.

    Returns
    -------
    list[Rect]
        Filtered boxes.
    """
    return list(
        filter(
            lambda box: box[2] >= char_shape[0] and box[3] >= char_shape[1], input_boxes
        )
    )


def merge_boxes(boxes: list[Rect]) -> Rect | None:
    """
    Merge multiple boxes into bounding box.

    Parameters
    ----------
    boxes : list[Rect]
        List of rectangles.

    Returns
    -------
    Rect | None
        Merged rectangle or None if empty.
    """
    if len(boxes) == 0:
        return None
    ls = [box[0] for box in boxes]
    us = [box[1] for box in boxes]
    rs = [box[0] + box[2] for box in boxes]
    ds = [box[1] + box[3] for box in boxes]
    bounds = [min(ls), min(us), max(rs), max(ds)]
    return [bounds[0], bounds[1], bounds[2] - bounds[0], bounds[3] - bounds[1]]


def _sorted_boxes_are_vertically_close(
    boxes: tuple[Rect, Rect], row_height: int
) -> bool:
    """boxes[0] is supposed to be above boxes[1]"""
    # _, y1, _, h1 = boxes[0]
    # y2 = boxes[1][1]
    # TODO: explain 1.5 factor
    return 0 <= box_y_center(boxes[1]) - box_y_center(boxes[0]) <= 1.5 * row_height


def box_y_center(box: Rect):
    """
    Calculate the y-center of a box.

    Parameters
    ----------
    box : Rect
        Rectangle.

    Returns
    -------
    int
        Y-center coordinate.
    """
    return int(box[1] + box[3] / 2)


def merge_same_line_boxes(boxes: list[Rect], line_height: int) -> list[Rect]:
    """Merge boxes that lie on the same line.

    Boxes are considered to be on the same line if their y-centers
    lie within char_height/2 from one another.

    Args:
        boxes: list of rectangles
        char_height: Character height used to determine line proximity

    Returns:
        List of merged boxes, sorted by y-coordinate
    """
    if not boxes:
        return []

    # Sort boxes by y-coordinate to process top-to-bottom
    boxes = sorted(boxes, key=lambda b: b[1])

    # Group boxes by their y-centers (boxes within char_height/2 belong to same line)
    merged_boxes: list[Rect] = []
    # TODO: explain line height threshold
    threshold = line_height / 2

    current_group: list[Rect] = [boxes[0]]
    current_y_center = box_y_center(current_group[0])

    for box in boxes[1:]:
        box_y_center_val = box_y_center(box)

        # Check if this box belongs to the current line
        if abs(box_y_center_val - current_y_center) <= threshold:
            current_group.append(box)
        else:
            # Merge the current group and start a new one
            merged_box = merge_boxes(current_group)
            if merged_box:
                merged_boxes.append(merged_box)
            current_group = [box]
            current_y_center = box_y_center(current_group[0])

    # Don't forget to merge the last group
    if current_group:
        merged_box = merge_boxes(current_group)
        if merged_box:
            merged_boxes.append(merged_box)

    return merged_boxes


def merge_and_filter_station_name_boxes(
    input_boxes: list[Rect],
    rows_positions: list[int],
    char_height: int,
    debug_img: MatLike | None = None,
) -> list[Rect]:
    """
    Merge boxes that are part of the same station name.

    Parts of a station name are generally within one table row of each other. Only the last box of a station name is aligned with the data row, though.

    Parameters
    ----------
    input_boxes : list[Rect]
        List of station name and basin boxes. Each box lies within a single row of the table.
    rows_positions : list[int]
        Row center y-coordinates.
    char_height : int
        Character height.
    debug_img : MatLike | None, optional
        Debug image to draw on.

    Returns
    -------
    list[Rect]
        Merged and filtered boxes.
    """
    input_boxes = sorted(input_boxes, key=lambda b: b[1])
    rows_heights = list(map(lambda p: p[1] - p[0], pairwise(rows_positions)))
    if not rows_heights:
        rows_heights = [char_height]
    # TODO: explain the 3 * char_height guess
    row_height = min(min(rows_heights), int(3 * char_height))

    horizontally_merged_boxes = sorted(
        merge_same_line_boxes(input_boxes, row_height), key=lambda b: b[1]
    )

    if debug_img is not None:
        _ = draw_bounding_boxes(
            debug_img,
            horizontally_merged_boxes,
            color=(255, 255, 0),
            overwrite=True,
        )

    rows_positions = sorted(rows_positions)

    output_boxes: list[Rect] = []

    merged_boxes_centers = np.array(
        [int(y + h / 2) for _, y, _, h in horizontally_merged_boxes]
    )
    last_matched_box_i = -1
    for row_position in rows_positions:
        matching_name_box_i = (
            int(
                np.argmin(
                    np.abs(
                        merged_boxes_centers[last_matched_box_i + 1 :] - row_position
                    )
                )
            )
            + last_matched_box_i
            + 1
        )
        to_merge_name_box_is = list(
            takewhile(
                lambda i: _sorted_boxes_are_vertically_close(
                    (horizontally_merged_boxes[i], horizontally_merged_boxes[i + 1]),
                    row_height,
                ),
                range(matching_name_box_i - 1, last_matched_box_i, -1),
            )
        )

        if to_merge_name_box_is:
            merged_box = merge_boxes(
                horizontally_merged_boxes[
                    to_merge_name_box_is[-1] : matching_name_box_i + 1
                ]
            )
            if not merged_box:
                logger.error("Merged box is None")
                merged_box = horizontally_merged_boxes[matching_name_box_i]
        else:
            merged_box = horizontally_merged_boxes[matching_name_box_i]
        last_matched_box_i = matching_name_box_i
        output_boxes.append(merged_box)

    return output_boxes


def detect_station_boxes(
    column_image: MatLike, char_shape: tuple[int, int], rows_centers: list[int]
) -> list[Rect]:
    """
    Detect station name boxes in first column.

    Parameters
    ----------
    column_image : MatLike
        First column image.
    char_shape : tuple[int, int]
        Character dimensions (width, height).
    rows_centers : list[int]
        Row center positions.

    Returns
    -------
    list[Rect]
        List of station box rectangles.
    """
    word_blobs = create_word_blobs(
        column_image,
        WordBlobsCreationConfiguration(gap_kernel_shape=(char_shape[0] // 2, 1)),
    )
    word_boxes = list(extract_contours_boxes(word_blobs))
    word_boxes.sort(key=lambda r: r[1])
    filtered_wboxes = filter_by_size(word_boxes, char_shape)
    filtered_wboxes = merge_and_filter_station_name_boxes(
        filtered_wboxes, rows_centers, char_shape[1]
    )
    return filtered_wboxes


def associate_closest_station_names(
    results: list[str | None], anagrafica: list[str]
) -> list[dict[str, Any]]:
    """
    Match OCR results to known station names using Levenshtein distance.

    Parameters
    ----------
    results : list[str | None]
        OCR recognized names.
    anagrafica : list[str]
        Known station names.

    Returns
    -------
    list[dict[str, Any]]
        List of dicts with matched names and similarity scores.
    """
    anagrafica_df = pl.DataFrame({"name_anagrafica": anagrafica}).with_columns(
        simplified_name=pl.col("name_anagrafica")
        .str.to_lowercase()
        .str.strip_chars()
        .str.replace_all(r"\s+", " ")
    )
    results_df = (
        (
            pl.DataFrame({"ocr_name": results}, schema={"ocr_name": pl.String})
            .with_columns(pl.row_index())
            .with_columns(
                simplified_name=pl.coalesce("ocr_name", pl.lit(""))
                .str.to_lowercase()
                .str.strip_chars()
                .str.replace_all(r"\s+", " ")
            )
            .join(anagrafica_df, how="cross", suffix="_anagrafica")
        )
        .with_columns(
            distance=(
                pl.col("simplified_name")  # pyright: ignore[reportUnknownMemberType]
                .dist_str.levenshtein("simplified_name_anagrafica")  # type: ignore  # pyright: ignore[reportAttributeAccessIssue]
                .cast(pl.Int32)
            )
        )
        .sort("index", "distance")
        .group_by("index", maintain_order=True)
        .head(1)
        .with_columns(similarity=-pl.col("distance"))
    )
    return (
        results_df.sort("index")
        .with_columns(
            name_anagrafica=pl.when(pl.col("ocr_name").is_null())
            .then("ocr_name")
            .otherwise("name_anagrafica")
        )
        .select(name="name_anagrafica", string_similarity="similarity")
        .to_dicts()
    )


def compute_name_rows(boxes: list[Rect]) -> list[int]:
    """
    Sort box indices by y-coordinate.

    Parameters
    ----------
    boxes : list[Rect]
        List of rectangles.

    Returns
    -------
    list[int]
        Sorted indices.
    """
    ys = [b[1] for b in boxes]
    return np.argsort(ys).tolist()


def transpose_recognition_results(
    recognition: list[dict[str, Any]],
) -> dict[str, list[Any]]:
    """
    Transpose recognition results to group by key.

    Parameters
    ----------
    recognition : list[dict[str, Any]]
        List of result dicts.

    Returns
    -------
    dict[str, list[Any]]
        Dict mapping keys to lists.
    """
    return {key: [r[key] for r in recognition] for key in recognition[0].keys()}


def merge_easypolys(polys: list[list[list[int]]] | NDArray[np.int_]) -> list[int]:
    """
    Merge multiple easyocr polygons into single rectangle.

    Parameters
    ----------
    polys : list[list[list[int]]] | NDArray[np.int_]
        List of polygons.

    Returns
    -------
    list[int]
        Merged rectangle [x1, x2, y1, y2].
    """
    polys = np.array(polys)
    return [
        int(polys[..., 0].min()),
        int(polys[..., 0].max()),
        int(polys[..., 1].min()),
        int(polys[..., 1].max()),
    ]


def process_easyocr_readtext_result(
    cell_results: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Process easyocr readtext results for a single cell.

    Parameters
    ----------
    cell_results : list[dict[str, Any]]
        List of recognition results.

    Returns
    -------
    dict[str, Any]
        Processed result with boxes, text, confidence.
    """
    return {
        "boxes": easyrect2rect(
            merge_easypolys([result["boxes"] for result in cell_results])
        )
        if cell_results
        else None,
        "text": " ".join([r["text"] for r in cell_results]) if cell_results else None,
        "confident": float(np.mean([r["confident"] for r in cell_results]))
        if cell_results
        else None,
    }


def process_easyocr_recognize_result(
    box_results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Process easyocr recognize results.

    Parameters
    ----------
    box_results : list[dict[str, Any]]
        List of box results.

    Returns
    -------
    list[dict[str, Any]]
        Processed results.
    """
    return [process_easyocr_readtext_result([box_result]) for box_result in box_results]


def easyocr_names_reader(ocr: easyocr.Reader):
    """
    Create a reader function for station names using EasyOCR.

    Parameters
    ----------
    ocr : easyocr.Reader
        Initialized EasyOCR reader.

    Returns
    -------
    Callable
        Reader function.
    """

    def _reader(
        image: MatLike, rois: list[Rect]
    ) -> tuple[list[str | None], list[float | None], list[Rect]]:
        recognition_results = transpose_recognition_results(
            [
                process_easyocr_readtext_result(
                    ocr.readtext(  # type: ignore  # pyright: ignore[reportUnknownMemberType, reportArgumentType]
                        extract(image, roi),
                        allowlist=f"{ascii_letters}()' /áàóòúùèéìi",
                        paragraph=False,
                        output_format="dict",
                    )
                )
                for roi in rois
            ]
        )
        recognition_results["boxes"] = [
            [box[0] + roi[0], box[1] + roi[1], box[2], box[3]] if box else roi
            for box, roi in zip(recognition_results["boxes"], rois)
        ]
        return (
            recognition_results["text"],
            recognition_results["confident"],
            recognition_results["boxes"],
        )

    return _reader


def easyocr_values_reader(ocr: easyocr.Reader):
    """
    Create a reader function for numerical values using EasyOCR.

    Parameters
    ----------
    ocr : easyocr.Reader
        Initialized EasyOCR reader.

    Returns
    -------
    Callable
        Reader function.
    """

    def _reader(
        image: MatLike, rois: list[Rect]
    ) -> tuple[list[str | None], list[float | None], list[Rect]]:
        easyrois = list(map(rect2easy, rois))
        recognition_results = transpose_recognition_results(
            process_easyocr_recognize_result(
                ocr.recognize(  # type: ignore  # pyright: ignore[reportUnknownMemberType, reportArgumentType]
                    image,
                    horizontal_list=easyrois,
                    free_list=[],
                    allowlist="_-0123456789",
                    output_format="dict",
                    batch_size=128,
                    sort_output=False,
                )
            )
        )
        return (
            recognition_results["text"],
            recognition_results["confident"],
            recognition_results["boxes"],
        )

    return _reader


# def paddleocr_values_reader(ocr: paddleocr.TextRecognition):
#     def _reader(
#         image: MatLike, rois: list[Rect]
#     ) -> tuple[list[str | None], list[float | None], list[Rect]]:
#         if image.ndim == 2:
#             image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
#         crops = list(
#             map(
#                 lambda roi: extract(
#                     image,
#                     roi,
#                 ),
#                 rois,
#             )
#         )
#         raw_results: list[dict[str, Any]] = ocr.predict(input=crops, batch_size=8)
#         return (
#             [r.get("rec_text") for r in raw_results],
#             [r.get("rec_score") for r in raw_results],
#             rois,
#         )
#     return _reader


def process_tesseract_cell_result(result: dict[str, Any], roi: Rect) -> dict[str, Any]:
    """
    Process Tesseract result for a single cell.

    Parameters
    ----------
    result : dict[str, Any]
        Tesseract result dict.
    roi : Rect
        ROI of the cell.

    Returns
    -------
    dict[str, Any]
        Processed result with boxes, confident, text.
    """
    text = " ".join(result["text"]).strip() if result["text"] else None
    if result["conf"]:
        confident = np.array(result["conf"])
        if (confident >= 0).any():
            confident = float(confident[confident >= 0].mean()) / 100
        else:
            confident = 0.0
    else:
        confident = None

    if len(result["left"]) > 0:
        i = min(len(result["left"]), 2) - 1
        boxes = [
            roi[0] + result["left"][i],
            roi[1] + result["top"][i],
            result["width"][i],
            result["height"][i],
        ]
    else:
        boxes = None
    return {"boxes": boxes, "confident": confident, "text": text}


def tesseract_values_reader(_: None):
    """
    Create a reader function for numerical values using Tesseract.

    Parameters
    ----------
    _ : None
        Unused parameter (for compatibility with other readers).

    Returns
    -------
    Callable
        Reader function.
    """

    def _reader(image: MatLike, rois: list[Rect]):
        crops = list(map(lambda roi: extract(image, roi), rois))
        recognition_results: tuple[list[str | None], list[float | None], list[Rect]] = (
            [],
            [],
            [],
        )
        for crop, roi in zip(crops, rois):
            processed_result = process_tesseract_cell_result(
                pytesseract.image_to_data(  # pyright: ignore[reportUnknownMemberType, reportArgumentType]
                    crop,
                    config="--psm 8 -c tessedit_char_whitelist=0123456789-_ --oem 1",
                    output_type="dict",
                ),
                roi,
            )
            recognition_results[0].append(processed_result["text"])
            recognition_results[1].append(processed_result["confident"])
            if processed_result["boxes"]:
                recognition_results[2].append(processed_result["boxes"])
            else:
                recognition_results[2].append(roi)

        return recognition_results

    return _reader


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
    """
    Read station names from the first column of the table.

    Parameters
    ----------
    image : MatLike
        The image of the table.
    rows_centers : list[int]
        The y-coordinates of the centers of the rows.
    column_separators : list[int]
        The x-coordinates of the column separators.
    char_shape : tuple[int, int]
        The approximate shape (width, height) of a character.
    reader : Callable[[MatLike, list[Rect]], tuple[list[str | None], list[float | None], list[Rect]]]
        A callable that takes an image and a list of ROIs and returns the recognized text, confidences, and bounding boxes.
    roi_padding : int
        The amount of padding to add around each ROI.
    anagrafica : list[str]
        A list of known station names to match against.

    Returns
    -------
    tuple[list[str | None], list[float | None], list[Rect]]
        A tuple containing:
        - A list of the matched station names from `anagrafica`.
        - A list of the similarity scores for the matches.
        - A list of the bounding boxes for the recognized names.

    Notes
    -----
    This function extracts the first column, detects text boxes, sorts them, reads them using the provided `reader`,
    associates the results with the closest names in `anagrafica`, and returns the results.
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


def populate_with_reading_results(
    results: tuple[list[str | None], list[float | None], list[Rect]],
    reader: Callable[
        [MatLike, list[Rect]], tuple[list[str | None], list[float | None], list[Rect]]
    ],
    image: MatLike,
    rois: list[Rect],
    threshold: float,
) -> None:
    """
    Fill the results tuple with OCR readings, only when the confidence is above the threshold.

    Parameters
    ----------
    results : tuple[list[str | None], list[float | None], list[Rect]]
        The results tuple to populate (readings, confidences, boxes).
    reader : Callable[[MatLike, list[Rect]], tuple[list[str | None], list[float | None], list[Rect]]]
        The OCR reader callable.
    image : MatLike
        The image to process.
    rois : list[Rect]
        The list of ROIs to read.
    threshold : float
        The confidence threshold. Results with confidence below this value are ignored.

    Notes
    -----
    This function filters out ROIs that already have satisfying results to avoid unnecessary OCR processing.
    It maps the new results back to their original positions in the `results` tuple.
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
    """
    Group values into batches of size n.
    """
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
    """
    Read values from the table cells using one or more OCR readers.

    Parameters
    ----------
    image : MatLike
        The image of the table.
    rows_centers : list[int]
        The y-coordinates of the centers of the rows.
    column_separators : list[int]
        The x-coordinates of the column separators.
    number_char_shape : tuple[int, int]
        The approximate shape (width, height) of a number character.
    readers : list[Callable[[MatLike, list[Rect]], tuple[list[str | None], list[float | None], list[Rect]]]]
        A list of OCR reader callables to attempt in order.
    confidence_thresholds : list[float]
        A list of confidence thresholds corresponding to each reader.
    roi_padding : int
        The amount of padding to add around each ROI.
    extra_width : int
        Extra width to add to the ROI.

    Returns
    -------
    tuple[list[list[str | None]], list[list[float | None]], list[list[Rect]]]
        A tuple containing:
        - A 2D list of the recognized values [col][row].
        - A 2D list of the confidences [col][row].
        - A 2D list of the bounding boxes [col][row].
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
                WordBlobsCreationConfiguration(gap_iterations=2, simple_iterations=0),
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

    return (
        _grouped(results_flat[0], n_rows),
        _grouped(results_flat[1], n_rows),
        _grouped(results_flat[2], n_rows),
    )
