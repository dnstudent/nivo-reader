from dataclasses import dataclass, field

from fancy_dataclass import TOMLDataclass, JSONDataclass
import cv2

from .preprocessing import LinesDetectionConfiguration, LinesCombinationConfiguration


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
