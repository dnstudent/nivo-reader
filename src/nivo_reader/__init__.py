"""NIVO table reader package."""

from .excel_output import save_artifacts, write_tables_to_excel
from .image_processing import ms_binary, ms_threshold, preproc
from .nivo_reader import read_nivo_table
from .ocr_processing import (
    associate_closest_station_names,
)
from .roi_utilities import autocrop, generate_roi_grid, pad_roi, roi_grid_coordinates
from .table_detection import (
    detect_column_separators,
    detect_lines,
    detect_rows_positions,
    remove_lines_from_image,
)

__all__ = [
    "read_nivo_table",
    "preproc",
    "ms_binary",
    "ms_threshold",
    "detect_lines",
    "remove_lines_from_image",
    "detect_rows_positions",
    "detect_column_separators",
    "associate_closest_station_names",
    "autocrop",
    "pad_roi",
    "generate_roi_grid",
    "roi_grid_coordinates",
    "write_tables_to_excel",
    "save_artifacts",
]
