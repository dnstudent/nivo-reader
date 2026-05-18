import logging
from dataclasses import dataclass, field
from typing import Any, cast, override

import cv2
import paddleocr
from cv2.typing import MatLike
from fancy_dataclass import TOMLDataclass

from nivo_reader.lib.common import BoundingBox
from nivo_reader.lib.images import extract
from nivo_reader.modules.table_digitization.base import (
    OCRCellSpec,
    OCRModule,
    OCRResult,
)

type EasyBbox = list[int]
type EasyFreeBox = list[list[int]]


logging.getLogger("paddlex").setLevel(logging.ERROR)
logging.getLogger("paddle").setLevel(logging.ERROR)


@dataclass
class PaddleOCR(
    TOMLDataclass,
    OCRModule[BoundingBox],
):
    model_name: str = "latin_PP-OCRv5_mobile_rec"
    kwargs: dict[str, Any] = field(default_factory=dict)
    name: str = "paddleocr"
    version: str = paddleocr.__version__

    @override
    def _preprocess_scan(self, scan: MatLike) -> MatLike:
        if scan.ndim == 2:
            scan = cv2.cvtColor(scan, cv2.COLOR_GRAY2BGR)
        return scan

    def __post_init__(self):
        self.reader: paddleocr.TextRecognition = paddleocr.TextRecognition(
            model_name=self.model_name, **self.kwargs
        )

    @override
    def _call(
        self, scan: MatLike, cells: list[OCRCellSpec[BoundingBox]], **kwargs: Any
    ):
        # print(scan.shape)
        crops = list(
            map(
                lambda cell: extract(
                    scan,
                    cell.bbox,
                ),
                cells,
            )
        )
        raw_results = self.reader.predict(input=crops, batch_size=128)  # pyright: ignore[reportUnknownMemberType]
        return [
            OCRResult(
                text=cast(str, r.get("rec_text")),
                confidence=cast(float, r.get("rec_score")),
                cell_spec=OCRCellSpec(
                    bbox=cell.bbox,
                    col=cell.col,
                    row=cell.row,
                ),
            )
            for r, cell in zip(raw_results, cells)
        ]
