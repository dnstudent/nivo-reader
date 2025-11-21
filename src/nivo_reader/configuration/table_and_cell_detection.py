from dataclasses import dataclass, field

from fancy_dataclass import TOMLDataclass, JSONDataclass
import cv2

from .preprocessing import LinesDetectionParameters, LinesCombinationParameters


@dataclass
class LinesExtractionParameters(TOMLDataclass, JSONDataclass):
    vertical_lines: LinesDetectionParameters = field(
        default_factory=lambda: LinesDetectionParameters.default("vertical")
    )
    horizontal_lines: LinesDetectionParameters = field(
        default_factory=lambda: LinesDetectionParameters.default("horizontal")
    )
    lines_combination: LinesCombinationParameters = field(
        default_factory=LinesCombinationParameters
    )


@dataclass
class DottedLinesRemovalParameters(TOMLDataclass, JSONDataclass):
    kernel_divisor: int = field(default=20)
    erosion_iterations: int = field(default=1)
    dilation_iterations: int = field(default=1)


@dataclass
class WordBlobsCreationParameters(TOMLDataclass, JSONDataclass):
    gap_kernel_type: int = field(
        default=cv2.MORPH_RECT
    )  #: cv2.MorphShapes = cv2.MORPH_RECT,
    gap_kernel_shape: tuple[int, int] = field(default=(6, 2))
    gap_iterations: int = field(default=5)
    simple_kernel_shape: tuple[int, int] = field(default=(3, 3))
    simple_iterations: int = field(default=1)


@dataclass
class TableProcessingParameters(TOMLDataclass, JSONDataclass):
    lines_removal: LinesExtractionParameters = field(
        default_factory=LinesExtractionParameters
    )
    word_blobs_creation: WordBlobsCreationParameters = field(
        default_factory=WordBlobsCreationParameters
    )
    dots_removal: DottedLinesRemovalParameters = field(
        default_factory=DottedLinesRemovalParameters
    )
