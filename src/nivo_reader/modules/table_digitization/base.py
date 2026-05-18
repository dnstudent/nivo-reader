from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast, override

import cv2
from cv2.typing import MatLike
from fancy_dataclass import JSONDataclass

from nivo_reader.lib.common import BoundingBox
from nivo_reader.modules.preprocessing.base import Preprocessor


@dataclass
class OCRCellSpec[B](JSONDataclass):
    bbox: B
    col: int
    row: int


@dataclass
class OCRResult[B](JSONDataclass):
    text: str
    confidence: float
    cell_spec: OCRCellSpec[B]


@dataclass
class OCRModule[B](ABC):
    name: str
    version: str

    def _preprocess_scan(self, scan: MatLike) -> MatLike:
        return scan

    def _preprocess_bbox(self, bbox: BoundingBox) -> B:
        return cast(B, bbox)

    def _postprocess_bbox(self, bbox: B) -> BoundingBox:
        return cast(BoundingBox, bbox)

    @abstractmethod
    def _call(
        self, scan: MatLike, cells: list[OCRCellSpec[B]], **kwargs: Any
    ) -> list[OCRResult[B]]:
        pass

    def _call_noconvert(
        self, scan: MatLike, cells: list[OCRCellSpec[BoundingBox]], **kwargs: Any
    ) -> list[OCRResult[B]]:
        ocr_cells = [
            OCRCellSpec[B](
                bbox=self._preprocess_bbox(cell.bbox),
                col=cell.col,
                row=cell.row,
            )
            for cell in cells
        ]
        scan = self._preprocess_scan(scan)
        return self._call(scan, ocr_cells, **kwargs)

    def __call__(
        self,
        scan: MatLike,
        cells: list[OCRCellSpec[BoundingBox]],
        **kwargs: Any,
    ) -> list[OCRResult[BoundingBox]]:
        results = self._call_noconvert(scan, cells, **kwargs)
        return [
            OCRResult(
                text=result.text,
                confidence=result.confidence,
                cell_spec=OCRCellSpec(
                    bbox=self._postprocess_bbox(result.cell_spec.bbox),
                    col=result.cell_spec.col,
                    row=result.cell_spec.row,
                ),
            )
            for result in results
        ]


@dataclass
class MultipleOCRModule(OCRModule[BoundingBox]):
    ocrs: tuple[OCRModule[Any], ...]
    filters: list[Callable[[OCRCellSpec[BoundingBox]], bool]]

    def __post_init__(self):
        assert len(self.ocrs) == len(self.filters), (
            "ocrs and filters must have the same length"
        )

    @override
    def _call(
        self, scan: MatLike, cells: list[OCRCellSpec[BoundingBox]], **kwargs: Any
    ) -> list[OCRResult[BoundingBox]]:
        cells_by_ocr = [list(filter(f, cells)) for f in self.filters]
        results: list[OCRResult[BoundingBox]] = []
        for ocr, cells in zip(self.ocrs, cells_by_ocr):
            results.extend(ocr(scan, cells, **kwargs))
        return results


class CellsListDigitizer[B]:
    def __init__(
        self,
        ocr: OCRModule[B],
        scan_processor: Preprocessor | None = None,
    ):
        self.ocr: OCRModule[B] = ocr
        self.scan_processor: Preprocessor | None = scan_processor

    def __call__(
        self,
        scan: MatLike,
        cells: list[OCRCellSpec[BoundingBox]],
        preprocessor_configuration: Mapping[str, Any] | None = None,
        debug_path: Path | None = None,
        **kwargs: Any,
    ):
        if self.scan_processor is not None:
            scan, _ = self.scan_processor(scan, preprocessor_configuration or {})
            if debug_path is not None:
                _ = cv2.imwrite(str(debug_path), scan)
        return self.ocr(scan, cells, **kwargs)
