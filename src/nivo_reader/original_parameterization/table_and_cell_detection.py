from typing import Self, Literal

from dataclasses import dataclass, field

from fancy_dataclass import TOMLDataclass, JSONDataclass
import cv2


@dataclass
class LinesDetectionParameters(TOMLDataclass, JSONDataclass):
    kernel_divisor: int = field(
        metadata={
            "doc": "Divisor of the image side to compute the convolution kernel side"
        }
    )
    erosion_iterations: int = field(
        default=1,
        metadata={"doc": "Iterations of the erosion procedure during line detection"},
    )
    dilation_iterations: int = field(
        default=5,
        metadata={"doc": "Iterations of the dilation procedure during line detection"},
    )

    @classmethod
    def default(cls, orientation: Literal["vertical", "horizontal"]) -> Self:
        if orientation == "horizontal":
            return cls(kernel_divisor=20)
        elif orientation == "vertical":
            return cls(kernel_divisor=50)
        else:
            raise AttributeError("Orientation must be 'horizontal' or 'vertical'")


@dataclass
class LinesCombinationParameters(TOMLDataclass, JSONDataclass):
    dilation_kernel_shape: tuple[int, int] = field(
        default=(2, 2),
        metadata={
            "doc": "Shape of the kernel for the dilation procedure in the lines combination operation"
        },
    )
    dilation_iterations: int = field(
        default=5,
        metadata={
            "doc": "Number of iterations of the dilation procedure in the lines combination operation"
        },
    )


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
