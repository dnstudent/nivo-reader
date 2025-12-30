from dataclasses import dataclass, field
from typing import Literal, Self

import cv2
from fancy_dataclass import JSONDataclass, TOMLDataclass


@dataclass
class LinesDetectionConfiguration(TOMLDataclass, JSONDataclass):
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
        assert orientation in ["vertical", "horizontal"]
        if orientation == "horizontal":
            return cls(kernel_divisor=20)
        if orientation == "vertical":
            return cls(kernel_divisor=50)


@dataclass
class AngleDetectionConfiguration(TOMLDataclass, JSONDataclass):
    lines_detection_configuration: LinesDetectionConfiguration = field(
        metadata={"doc": "Configuration for line detection"}
    )
    line_ratio_threshold: float = field(
        default=20,
        metadata={
            "doc": "Threshold for the height-to-width or width-to-height of the bounding box of a contour to consider it a line"
        },
    )

    @classmethod
    def default(cls, orientation: Literal["vertical", "horizontal"]) -> Self:
        return cls(
            lines_detection_configuration=LinesDetectionConfiguration.default(
                orientation=orientation
            ),
            line_ratio_threshold=20,
        )


@dataclass
class LinesCombinationConfiguration(TOMLDataclass, JSONDataclass):
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
class BinarizationConfiguration(TOMLDataclass, JSONDataclass):
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


@dataclass
class ThresholdConfiguration(TOMLDataclass, JSONDataclass):
    kernel_shape: tuple[int, int] = field(
        default=(5, 5), metadata={"doc": "Shape of the kernel used for thresholding"}
    )
    kernel_type: int = field(
        default=cv2.MORPH_RECT,
        metadata={"doc": "Morphology of the threshold filling kernel"},
    )


@dataclass
class PreprocessConfiguration(TOMLDataclass, JSONDataclass):
    binarization_configuration: BinarizationConfiguration = field(
        default_factory=BinarizationConfiguration
    )
    threshold_configuration: ThresholdConfiguration = field(
        default_factory=ThresholdConfiguration
    )
    angle_detection_configuration: AngleDetectionConfiguration = field(
        default_factory=lambda: AngleDetectionConfiguration.default("vertical")
    )
