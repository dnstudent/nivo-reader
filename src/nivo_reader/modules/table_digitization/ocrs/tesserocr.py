"""nivo-reader interface to Tesserocr"""

import os
import re
from dataclasses import dataclass, field
from typing import Any, override

import cv2
import tesserocr
from cv2.typing import MatLike
from fancy_dataclass import TOMLDataclass
from PIL import Image

from nivo_reader.lib.common import BoundingBox
from nivo_reader.modules.table_digitization.base import (
    OCRCellSpec,
    OCRModule,
    OCRResult,
)


def _read_tessversion() -> str:
    full_line = tesserocr.tesseract_version()
    match = next(re.finditer(r"tesseract\s*(\d+\.\d+\.\d+)", full_line), None)
    return match.group(1) if match else ""


@dataclass
class TesserocrOCR(OCRModule[BoundingBox], TOMLDataclass):
    """OCR Module using Tesserocr library"""

    lang: str = "ita"
    oem: tesserocr.OEM = tesserocr.OEM.DEFAULT
    psm: tesserocr.PSM = tesserocr.PSM.AUTO
    variables: dict[str, str] = field(default_factory=dict)
    tessdata_dir: str | None = None
    name: str = "tesserocr"
    version: str = _read_tessversion()

    def _create_api(self) -> tesserocr.PyTessBaseAPI:
        """Create a new Tesserocr API instance with configured parameters"""
        if self.tessdata_dir is None:
            self.tessdata_dir = os.environ["TESSDATA_PREFIX"]
        return tesserocr.PyTessBaseAPI(
            self.tessdata_dir,
            lang=self.lang,
            oem=self.oem,
            psm=self.psm,
            variables=self.variables,
        )

    @override
    def _call(
        self, scan: MatLike, cells: list[OCRCellSpec[BoundingBox]], **kwargs: Any
    ) -> list[OCRResult[BoundingBox]]:
        results: list[OCRResult[BoundingBox]] = []

        with self._create_api() as api:
            # Set the full image once
            img = cv2.cvtColor(scan, cv2.COLOR_BGR2RGB)
            api.SetImage(Image.fromarray(img))  # pyright: ignore[reportUnknownMemberType]

            for cell in cells:
                # Set the rectangle for this specific cell
                api.SetRectangle(
                    cell.bbox.x, cell.bbox.y, cell.bbox.width, cell.bbox.height
                )

                # Get OCR result
                confidence = api.AllWordConfidences()
                text = api.GetUTF8Text().strip()

                # Calculate average confidence
                avg_confidence: float = (
                    float(sum(confidence)) / len(confidence) / 100
                    if confidence
                    else 0.0
                )

                results.append(
                    OCRResult(
                        text=text,
                        confidence=avg_confidence,
                        cell_spec=OCRCellSpec(
                            bbox=cell.bbox, col=cell.col, row=cell.row
                        ),
                    )
                )

        return results


def build_tesserocr_number_ocr(
    lang: str, extra_whitelist: str, tessdata_dir: str | None = None
) -> TesserocrOCR:
    return TesserocrOCR(
        tessdata_dir=tessdata_dir,
        lang=lang,
        oem=tesserocr.OEM.LSTM_ONLY,
        psm=tesserocr.PSM.SINGLE_WORD,
        variables={"tessedit_char_whitelist": f"0123456789{extra_whitelist}"},
    )


def build_tesserocr_text_ocr(
    lang: str, tessdata_dir: str | None = None
) -> TesserocrOCR:
    return TesserocrOCR(
        tessdata_dir=tessdata_dir,
        lang=lang,
        oem=tesserocr.OEM.LSTM_ONLY,
        psm=tesserocr.PSM.SINGLE_BLOCK,
    )
