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
along with this program.  If not, see <https://www.gnu.org/licenses/>."""

from dataclasses import dataclass, field

import cv2
from fancy_dataclass import JSONDataclass, TOMLDataclass

from .preprocessing import LinesCombinationConfiguration, LinesDetectionConfiguration


@dataclass
class LinesExtractionConfiguration(TOMLDataclass, JSONDataclass):
    vertical_lines: LinesDetectionConfiguration = field(
        default_factory=lambda: LinesDetectionConfiguration.default("vertical")
    )
    horizontal_lines: LinesDetectionConfiguration = field(
        default_factory=lambda: LinesDetectionConfiguration.default("horizontal")
    )
    lines_combination: LinesCombinationConfiguration = field(
        default_factory=LinesCombinationConfiguration
    )


@dataclass
class DottedLinesRemovalConfiguration(TOMLDataclass, JSONDataclass):
    kernel_divisor: int = field(default=20)
    erosion_iterations: int = field(default=1)
    dilation_iterations: int = field(default=1)


@dataclass
class WordBlobsCreationConfiguration(TOMLDataclass, JSONDataclass):
    gap_kernel_type: int = field(
        default=cv2.MORPH_RECT
    )  #: cv2.MorphShapes = cv2.MORPH_RECT,
    gap_kernel_shape: tuple[int, int] = field(default=(6, 2))
    gap_iterations: int = field(default=5)
    simple_kernel_shape: tuple[int, int] = field(default=(3, 3))
    simple_iterations: int = field(default=1)


@dataclass
class TableProcessingConfiguration(TOMLDataclass, JSONDataclass):
    lines_removal: LinesExtractionConfiguration = field(
        default_factory=LinesExtractionConfiguration
    )
    word_blobs_creation: WordBlobsCreationConfiguration = field(
        default_factory=WordBlobsCreationConfiguration
    )
    dots_removal: DottedLinesRemovalConfiguration = field(
        default_factory=DottedLinesRemovalConfiguration
    )
