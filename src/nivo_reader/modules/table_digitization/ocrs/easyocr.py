"""nivo-reader interface to EasyOCR"""

from abc import ABC
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast, override

import easyocr
import numpy as np
from cv2.typing import MatLike
from fancy_dataclass import TOMLDataclass
from fancy_dataclass.json import JSONDataclass
from numpy.typing import NDArray

from nivo_reader.lib.common import BoundingBox
from nivo_reader.modules.table_digitization.base import (
    OCRCellSpec,
    OCRModule,
    OCRResult,
)

type EasyBbox = list[int]
type EasyFreeBox = list[list[int]]


@dataclass
class EasyOCR(OCRModule[EasyBbox], TOMLDataclass, ABC):
    call_config: dict[str, str | int | float | bool]
    lang_list: list[str]
    verbose: bool = False
    quantize: bool = False
    gpu: bool = True
    model_storage_directory: Path | None = None
    user_network_directory: Path | None = None
    detect_network: str = "craft"
    recog_network: str = "standard"
    download_enabled: bool = True
    detector: bool = True
    recognizer: bool = True
    cudnn_benchmark: bool = False

    def __post_init__(self):
        self.reader: easyocr.Reader = easyocr.Reader(
            self.lang_list,
            gpu=self.gpu,
            model_storage_directory=self.model_storage_directory,
            user_network_directory=self.user_network_directory,
            detect_network=self.detect_network,
            recog_network=self.recog_network,
            download_enabled=self.download_enabled,
            detector=self.detector,
            recognizer=self.recognizer,
            verbose=self.verbose,
            quantize=self.quantize,
            cudnn_benchmark=self.cudnn_benchmark,
        )

    @override
    @classmethod
    def _preprocess_bbox(cls, bbox: BoundingBox) -> EasyBbox:
        return [bbox.x, bbox.x + bbox.width, bbox.y, bbox.y + bbox.height]

    @override
    @classmethod
    def _postprocess_bbox(cls, bbox: EasyBbox) -> BoundingBox:
        x1, x2, y1, y2 = bbox
        return BoundingBox(x1, y1, x2 - x1, y2 - y1)

    @classmethod
    def _freebox_to_easybox(cls, freebox: EasyFreeBox | None) -> EasyBbox:
        if not freebox:
            return [0, 0, 0, 0]
        if len(freebox) != 4:
            raise ValueError(f"Invalid freebox: {freebox}")
        x1, y1 = freebox[0]
        x2, y2 = freebox[2]
        return [x1, x2, y1, y2]


@dataclass
class EasyOCRWord(EasyOCR, TOMLDataclass):
    @override
    def _call(
        self, scan: MatLike, cells: list[OCRCellSpec[EasyBbox]], **kwargs: Any
    ) -> list[OCRResult[EasyBbox]]:
        bboxes = [cell.bbox for cell in cells]
        _results = cast(
            list[dict[str, Any]],
            self.reader.recognize(  # pyright: ignore[reportUnknownMemberType]
                scan,
                horizontal_list=bboxes,
                free_list=[],
                detail=1,
                paragraph=False,
                output_format="dict",
                sort_output=False,
                **self.call_config,
                **kwargs,
            ),
        )
        return [
            OCRResult(
                text=cast(str, r["text"]),
                confidence=cast(float, r["confident"]),
                cell_spec=OCRCellSpec(
                    bbox=self._freebox_to_easybox(r["boxes"]),
                    col=cell.col,
                    row=cell.row,
                ),
            )
            for r, cell in zip(_results, cells)
        ]


@dataclass
class EasyOCRParagraph(EasyOCR, TOMLDataclass):
    @dataclass
    class ReadTextResult(JSONDataclass):
        text: str
        confident: float
        boxes: EasyFreeBox | None

        def to_ocrresult(self, cell: OCRCellSpec[EasyBbox]) -> OCRResult[EasyBbox]:
            valid_box = (
                cell.bbox
                if self.boxes is None
                else EasyOCRParagraph._freebox_to_easybox(self.boxes)  # pyright: ignore[reportPrivateUsage]
            )
            return OCRResult(
                text=self.text,
                confidence=self.confident,
                cell_spec=OCRCellSpec(
                    bbox=valid_box,
                    col=cell.col,
                    row=cell.row,
                ),
            )

    @classmethod
    def merge_easypolys(
        cls, polys: list[EasyFreeBox] | NDArray[np.int_]
    ) -> EasyFreeBox | None:
        """
        Merge multiple easyocr polygons into single rectangle.

        Parameters
        ----------
        polys : list[list[list[int]]] | NDArray[np.int_]
            List of polygons.

        Returns
        -------
        list[int]
            Merged rectangle [[x1,y1], [x2,y1], [x2, y2], [x1, y2]].
        """
        if len(polys) == 0:
            return None
        polys = np.array(polys)
        x1 = int(polys[..., 0].min())
        x2 = int(polys[..., 0].max())
        y1 = int(polys[..., 1].min())
        y2 = int(polys[..., 1].max())
        return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]

    @classmethod
    def extract(cls, image: MatLike, rect: EasyBbox) -> MatLike:
        """
        Extract rectangular region from image.

        Parameters
        ----------
        image : MatLike
            Input image.
        rect : Rect
            Rectangle (x1, x2, y1, y2).

        Returns
        -------
        MatLike
            Extracted region.
        """
        x1, x2, y1, y2 = rect
        return image[
            max(y1, 0) : min(y2, image.shape[0]),
            max(x1, 0) : min(x2, image.shape[1]),
        ]

    @classmethod
    def merge_readtext_result(cls, result: list[dict[str, Any]]) -> ReadTextResult:
        return EasyOCRParagraph.ReadTextResult(
            text=" ".join([r["text"] for r in result]),
            confident=float(
                np.mean([r["confident"] for r in result]) if result else 0.0
            ),
            boxes=cls.merge_easypolys([r["boxes"] for r in result]),
        )

    @override
    def _call(
        self, scan: MatLike, cells: list[OCRCellSpec[EasyBbox]], **kwargs: Any
    ) -> list[OCRResult[EasyBbox]]:
        bboxes = [cell.bbox for cell in cells]

        _results: list[EasyOCRParagraph.ReadTextResult] = []
        for bbox in bboxes:
            reading = cast(
                list[dict[str, Any]],
                self.reader.readtext(  # pyright: ignore[reportUnknownMemberType]
                    self.extract(scan, bbox),
                    paragraph=False,
                    output_format="dict",
                    **self.call_config,
                    **kwargs,
                ),
            )
            merged = self.merge_readtext_result(reading)
            _results.append(merged)
        return [r.to_ocrresult(cell) for r, cell in zip(_results, cells)]
