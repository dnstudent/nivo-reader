from dataclasses import dataclass, field
from typing import Literal, Self

import cv2
from fancy_dataclass import JSONDataclass, TOMLDataclass


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
class BinarizationParameters(TOMLDataclass, JSONDataclass):
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
class ThresholdParameters(TOMLDataclass, JSONDataclass):
    kernel_shape: tuple[int, int] = field(
        default=(5, 5), metadata={"doc": "Shape of the kernel used for thresholding"}
    )
    kernel_type: int = field(
        default=cv2.MORPH_RECT,
        metadata={"doc": "Morphology of the threshold filling kernel"},
    )


@dataclass
class PreprocessingParameters(TOMLDataclass, JSONDataclass):
    binarization_parameters: BinarizationParameters = field(
        default_factory=BinarizationParameters
    )
    threshold_parameters: ThresholdParameters = field(
        default_factory=ThresholdParameters
    )
