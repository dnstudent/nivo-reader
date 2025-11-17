"""NIVO table reader package."""

from .nivo_reader import read_nivo_table
from .image_processing import preproc, ms_binary, ms_threshold
from .table_detection import (
    detect_lines,
    remove_lines_from_image,
    detect_rows_positions,
    detect_column_separators,
)
from .ocr_processing import (
    associate_closest_station_names,
    autocrop,
    pad_roi,
)
from .roi_utilities import generate_roi_grid, roi_grid_coordinates
from .excel_output import write_tables_to_excel, save_artifacts

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
