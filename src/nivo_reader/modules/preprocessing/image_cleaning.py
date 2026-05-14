from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, final, override

import cv2
from cv2.typing import MatLike
from fancy_dataclass import TOMLDataclass

from nivo_reader.configuration.preprocessing import (
    LinesCombinationConfiguration,
    LinesDetectionConfiguration,
)
from nivo_reader.configuration.table_and_cell_detection import (
    LinesExtractionConfiguration,
)
from nivo_reader.lib.common import BoundingBox
from nivo_reader.table_detection import remove_lines_from_image

from .base import Preprocessor


@dataclass
class Cropping(Preprocessor, TOMLDataclass):
    bbox: BoundingBox

    @override
    def __call__(
        self, image: MatLike, configuration: Mapping[str, Any], **kwargs: Any
    ) -> tuple[MatLike, dict[str, Any]]:
        return image[
            self.bbox.y : self.bbox.y + self.bbox.height,
            self.bbox.x : self.bbox.x + self.bbox.width,
        ], {}


@final
@dataclass
class Binarization(Preprocessor, TOMLDataclass):
    name = "binarization"
    adaptive_threshold_type: int = field(
        default=cv2.ADAPTIVE_THRESH_MEAN_C,
        metadata={
            "doc": "Kind of adaptive threshold type for image binarization from python-opencv"
        },
    )
    region_side: int = field(
        default=91, metadata={"doc": "Side if the adaptive threshold region"}
    )
    threshold_c: int = field(
        default=6, metadata={"doc": "c value of the adaptive threshold algorithm"}
    )

    @override
    def __call__(
        self, image: MatLike, configuration: Mapping[str, Any], **kwargs: Any
    ) -> tuple[MatLike, dict[str, Any]]:
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return cv2.adaptiveThreshold(
            image,
            255,
            self.adaptive_threshold_type,
            cv2.THRESH_BINARY,
            self.region_side,
            self.threshold_c,
        ), {}


@final
@dataclass
class LineEraser(Preprocessor, TOMLDataclass):
    name = "line_eraser"
    vertical_lines_detection: LinesDetectionConfiguration = field(
        default_factory=lambda: LinesDetectionConfiguration.default("vertical")
    )
    horizontal_lines_detection: LinesDetectionConfiguration = field(
        default_factory=lambda: LinesDetectionConfiguration.default("horizontal")
    )
    lines_combination: LinesCombinationConfiguration = field(
        default_factory=LinesCombinationConfiguration
    )

    @override
    def __call__(
        self, image: MatLike, configuration: Mapping[str, Any], **kwargs: Any
    ) -> tuple[MatLike, dict[str, Any]]:

        return remove_lines_from_image(
            255 - image,
            LinesExtractionConfiguration(
                self.vertical_lines_detection,
                self.horizontal_lines_detection,
                self.lines_combination,
            ),
        ), {}
