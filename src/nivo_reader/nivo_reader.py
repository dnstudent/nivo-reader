"""Main NIVO table reader module."""

from pathlib import Path

import easyocr
from cv2.typing import MatLike


from . import ocr_processing
from .configuration.preprocessing import (
    PreprocessConfiguration,
    ThresholdConfiguration,
)
from .configuration.table_and_cell_detection import (
    LinesExtractionConfiguration,
)
from .excel_output import (
    draw_bounding_boxes,
    draw_straight_lines,
    save_artifacts,
    write_tables_to_excel,
)
from .image_processing import preprocess
from .ocr_processing import (
    easyocr_names_reader,
    read_station_names,
    read_values,
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
    """
    Extract NIVO table data from image and write to Excel.

    Parameters
    ----------
    original_image : MatLike
        The input image containing the table.
    excel_out_path : Path | str
        The path where the output Excel file will be saved.
    clips : tuple[int, int, int, int]
        Clipping margins (up, down, left, right) to apply to the table region.
    table_shape : tuple[int, int]
        The expected (width, height) of the table in the image.
    anagrafica : list[str]
        A list of known station names used for matching and validation.
    easyreader : easyocr.Reader
        An initialized EasyOCR reader instance.
    station_char_shape : tuple[int, int], optional
        The approximate shape (width, height) of characters in station names. Default is (12, 10).
    number_char_shape : tuple[int, int], optional
        The approximate shape (width, height) of number characters in the table. Default is (12, 20).
    roi_padding : int, optional
        The padding defined in pixels to add around each region of interest. Default is 3.
    nchars_threshold : int, optional
        The minimum number of characters required to detect a row or column. Default is 30.
    extra_width : int, optional
        Extra width to add to cell ROIs. Default is 6.
    low_confidence_threshold : float, optional
        The confidence threshold below which a cell reading is considered low confidence. Default is 0.7.
    debug_dir : Path | None, optional
        Directory where debug artifacts (images, intermediate steps) will be saved. If None, no debug artifacts are saved.

    Raises
    ------
    ValueError
        If the table rectangle cannot be detected in the image.
    """
    if debug_dir:
        debug_dir = Path(debug_dir)
        debug_dir.mkdir(parents=True, exist_ok=True)
        debug_image = original_image.copy()

    # Preprocessing
    image, threshold_image, binarized_image, _ = preprocess(
        original_image,
        deskew_method="nivo",
        configuration=PreprocessConfiguration(),
    )

    # Detect table rectangle
    rect = try_detect_table_rect(image, table_shape, ThresholdConfiguration())

    if rect is None:
        raise ValueError("Could not detect table rectangle in image")

    # Cut out table
    _, binarized_subtable = cut_out_tables(binarized_image, rect, clips)
    _, threshold_subtable = cut_out_tables(threshold_image, rect, clips)

    # Remove lines
    binarized_subtable_wo_lines = remove_lines_from_image(
        255 - binarized_subtable,
        LinesExtractionConfiguration(),
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
                "01_table_rect": draw_bounding_boxes(debug_image, [rect]),  # pyright: ignore[reportPossiblyUnboundVariable]
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
