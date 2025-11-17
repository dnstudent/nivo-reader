from dataclasses import dataclass, field

import cv2
from fancy_dataclass import TOMLDataclass, JSONDataclass


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
