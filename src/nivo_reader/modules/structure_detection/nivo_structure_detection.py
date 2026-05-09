"""
nivo-reader: a tool to digitize snowfall data tables from the Italian Hydrological Service
Copyright (C) 2026  Davide Nicoli

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

from dataclasses import dataclass
from typing import Any, final, override
from collections.abc import Mapping
import logging

import cv2
from cv2.typing import MatLike

from .base import (
    StructureDetector,
    StructureResult,
    TableSpec,
    CellSpec,
    HeaderSpec,
    IndexSpec,
    ContentSpec,
)
from nivo_reader.lib.common import BoundingBox, RectShape, ClipSizes
from nivo_reader.configuration.preprocessing import (
    ThresholdConfiguration,
    BinarizationConfiguration,
)
from nivo_reader.configuration.table_and_cell_detection import (
    LinesExtractionConfiguration,
    WordBlobsCreationConfiguration,
)
from nivo_reader.table_detection import (
    cut_out_tables,
    remove_lines_from_image,
    detect_rows_positions,
    detect_column_separators,
)
from .table_detection.nivo_table_detection import NivoTableDetection
from nivo_reader.image_processing import ms_threshold, ms_binary
from nivo_reader.ocr_processing import detect_station_boxes
from nivo_reader.roi_utilities import (
    generate_roi_grid,
    prepare_value_roi,
    pad_roi,
    autocrop_roi,
)
from pydantic import BaseModel, NonNegativeInt, PositiveInt


class RectSpec(BaseModel):
    width: PositiveInt
    height: PositiveInt

    def to_rect_shape(self) -> RectShape:
        return RectShape(width=self.width, height=self.height)


class BoundingBoxSpec(BaseModel):
    x: NonNegativeInt
    y: NonNegativeInt
    width: PositiveInt
    height: PositiveInt

    def to_bbox(self) -> BoundingBox:
        return BoundingBox(x=self.x, y=self.y, width=self.width, height=self.height)


class ClipSpec(BaseModel):
    top: NonNegativeInt
    bottom: NonNegativeInt
    left: NonNegativeInt
    right: NonNegativeInt

    def to_clip_sizes(self) -> ClipSizes:
        return ClipSizes(
            top=self.top, bottom=self.bottom, left=self.left, right=self.right
        )


class NivoStructureDetectionConfig(BaseModel):
    table_shape: RectSpec
    clips: ClipSpec
    station_char_shape: RectSpec
    number_char_shape: RectSpec
    roi_padding: NonNegativeInt
    nchars_threshold: NonNegativeInt
    extra_width: NonNegativeInt
    multi_row_station_names: bool
    from_extracted_structure: bool
    forced_bbox: BoundingBoxSpec | None = None


logger = logging.getLogger(__name__)


@final
@dataclass
class NivoStructureDetection(StructureDetector):
    name: str = "nivo_structure_detection"

    @override
    def __call__(
        self,
        image: MatLike,
        configuration: Mapping[str, Any] | NivoStructureDetectionConfig,
        previous_work: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> tuple[StructureResult, dict[str, Any]]:
        # Extract configuration
        if isinstance(configuration, Mapping):
            conf = NivoStructureDetectionConfig(**configuration)
        else:
            conf = configuration

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image

        if conf.forced_bbox is not None:
            table_bbox = conf.forced_bbox.to_bbox()
        else:
            # Detect table rectangle
            detector = NivoTableDetection(
                "detector",
                conf.table_shape.to_rect_shape(),
                ThresholdConfiguration(),
                conf.from_extracted_structure,
            )
            detection_result = detector(gray)[0]
            table_bbox = None if detection_result is None else detection_result[0]

        if table_bbox is None:
            raise ValueError(
                f"Could not detect table rectangle in image with {configuration}"
            )

        # Cut out table
        # Apply thresholding. This is done before cutting because
        # the dimensional parameters are determined on the original image.
        # NOTE: does not seem very smart.
        thresh = ms_threshold(gray, ThresholdConfiguration())
        binarized_image = ms_binary(gray, BinarizationConfiguration())

        clips = conf.clips.to_clip_sizes()
        # _, binarized_content_region = cut_out_tables(binarized_image, table_bbox, clips)
        _, threshold_content_region = cut_out_tables(thresh, table_bbox, clips)

        binarized_image_wo_lines = remove_lines_from_image(
            255 - binarized_image, LinesExtractionConfiguration()
        )
        _, binarized_content_wo_lines = cut_out_tables(
            binarized_image_wo_lines, table_bbox, clips
        )

        # rows centers are relative to clips.top + table_bbox.y
        rows_centers = detect_rows_positions(
            binarized_content_wo_lines,
            conf.nchars_threshold,
            conf.number_char_shape.to_rect_shape(),
        ).tolist()
        # column separators are relative to clips.left + table_bbox.x
        cols_separators = detect_column_separators(
            threshold_content_region, conf.number_char_shape.width
        )

        cells: list[CellSpec] = []

        # Station names
        first_column = binarized_content_wo_lines[
            :, cols_separators[0] : cols_separators[1]
        ]
        station_boxes_relative = detect_station_boxes(
            first_column,
            conf.station_char_shape.to_rect_shape(),
            rows_centers,
            conf.multi_row_station_names,
        )

        for row_idx, box in enumerate(station_boxes_relative):
            abs_box = BoundingBox(
                x=box.x + table_bbox.x + clips.left + cols_separators[0],
                y=box.y + table_bbox.y + clips.top,
                width=box.width,
                height=box.height,
            )
            padded_boxes = pad_roi(
                autocrop_roi(abs_box, binarized_image_wo_lines), conf.roi_padding
            )
            cells.append(
                CellSpec(
                    row=row_idx,
                    column=0,
                    cell_region=padded_boxes,
                )
            )

        # Value cells
        rois_grid = generate_roi_grid(
            rows_centers,
            cols_separators[1:],
            conf.number_char_shape.height,
            conf.extra_width,
        )

        for row_idx, _ in enumerate(rows_centers):
            for col_idx in range(len(cols_separators) - 2):
                relative_roi = rois_grid[col_idx][row_idx]
                abs_roi = BoundingBox(
                    x=relative_roi.x + table_bbox.x + clips.left,
                    y=relative_roi.y + table_bbox.y + clips.top,
                    width=relative_roi.width,
                    height=relative_roi.height,
                )

                prepared_rois = prepare_value_roi(
                    abs_roi,
                    binarized_image_wo_lines,
                    conf.number_char_shape.to_rect_shape(),
                    WordBlobsCreationConfiguration(
                        gap_iterations=2, simple_iterations=0
                    ),
                    conf.roi_padding,
                )
                cells.append(
                    CellSpec(
                        row=row_idx,
                        column=col_idx + 1,
                        cell_region=prepared_rois,
                    )
                )

        header_spec = HeaderSpec(header_region=table_bbox.top_slice(clips.top))
        index_spec = IndexSpec(
            index_region=BoundingBox(
                x=table_bbox.x,
                y=table_bbox.y + clips.top,
                width=clips.left,
                height=table_bbox.height - clips.top - clips.bottom,
            )
        )

        content_spec = ContentSpec(
            content_region=table_bbox.cut_clips(clips),
            cells=cells,
            nrows=len(rows_centers),
            ncols=len(cols_separators) - 1,
        )

        table_spec = TableSpec(
            content_spec=content_spec,
            header_spec=header_spec,
            index_spec=index_spec,
            full_region=table_bbox,
        )

        results = StructureResult(tables=[table_spec])

        return results, previous_work or {}
