# TODO: credit meteosaver authors
"""
nivo-reader: a tool to automatically read meteorological data
Copyright (C) 2026  Davide Nicoli

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

from dataclasses import dataclass
import logging
from typing import final, override, Any

from fancy_dataclass import JSONDataclass
import cv2
from cv2.typing import MatLike
import numpy as np

from nivo_reader.configuration.preprocessing import ThresholdConfiguration
from nivo_reader.image_processing import my_table_struct
from nivo_reader.lib.common import BoundingBox, RectShape
from .base import TableDetection


@final
@dataclass
class NivoTableDetection(TableDetection, JSONDataclass):
    expected_table_shape: RectShape
    threshold_configuration: ThresholdConfiguration
    from_extracted_structure: bool

    @override
    def __call__(
        self, image: MatLike, previous_work: dict[str, Any] | None = None
    ) -> tuple[list[BoundingBox] | None, dict[str, Any]]:
        if previous_work and "_gray_image" in previous_work:
            gray_image = previous_work["_gray_image"]
        elif image.ndim == 2:
            gray_image = image
        else:
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        try:
            rect = try_detect_table_rect(
                gray_image,
                self.expected_table_shape,
                self.threshold_configuration,
                self.from_extracted_structure,
            )
        except Exception:
            logging.exception(
                f"Could not detect table rectangle in image with {self.threshold_configuration} and expected shape {self.expected_table_shape} and from_extracted_structure {self.from_extracted_structure}"
            )
            rect = None

        if rect is None:
            try:
                rect = try_detect_table_rect(
                    gray_image,
                    self.expected_table_shape,
                    self.threshold_configuration,
                    not self.from_extracted_structure,
                )
            except Exception:
                logging.exception(
                    f"Could not detect table rectangle in image with {self.threshold_configuration} and expected shape {self.expected_table_shape} and from_extracted_structure {not self.from_extracted_structure}"
                )
                rect = None

        if rect is None:
            otherdetect = ParmaNivoTableDetection("")
            return otherdetect(image)

        return [rect], previous_work or {}


@final
@dataclass
class ParmaNivoTableDetection(TableDetection, JSONDataclass):
    @staticmethod
    def bbox2bbox(bbox: Any) -> BoundingBox:
        return BoundingBox(
            x=bbox.x1, y=bbox.y1, width=bbox.x2 - bbox.x1, height=bbox.y2 - bbox.y1
        )

    @override
    def __call__(
        self, image: MatLike, previous_work: dict[str, Any] | None = None
    ) -> tuple[list[BoundingBox] | None, dict[str, Any]]:
        from img2table.document import Image

        img = Image(cv2.imencode(".png", image)[1].tobytes(), detect_rotation=False)
        tables = img.extract_tables(implicit_rows=True)
        if len(tables) > 0:
            return list(
                map(lambda t: self.bbox2bbox(t.bbox), tables)
            ), previous_work or {}
        else:
            raise ValueError("Could not detect table rectangle")


def try_detect_table_rect(
    gray_image: MatLike,
    expected_shape: RectShape,
    threshold_configuration: ThresholdConfiguration,
    from_extracted_structure: bool,
) -> BoundingBox | None:
    """
    Detect table rectangle in image.

    Parameters
    ----------
    gray_image : MatLike
        Grayscale image.
    expected_table_shape : RectShape
        Expected table (width, height).
    threshold_configuration : ThresholdConfiguration
        Threshold configuration.
    from_extracted_structure : bool
        First remove content, then detect borders
    Returns
    -------
    Rect | None
        Bounding rectangle of table or None if not found.
    """
    thresh = ms_threshold(gray_image, threshold_configuration)
    if from_extracted_structure:
        thresh = my_table_struct(thresh)
    bboxes = sorted(
        filter(
            lambda r: (
                within_tolerance(r[2], expected_shape.width, tol=0.05)
                and within_tolerance(r[3], expected_shape.height, tol=0.05)
            ),
            map(
                lambda cnt: cv2.boundingRect(cnt),
                cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0],
            ),
        ),
        key=lambda r: abs(expected_shape.width * expected_shape.height - r[2] * r[3]),
    )

    if len(bboxes) > 0:
        bbox = BoundingBox.from_rect(bboxes[0])
        bbox.x -= 3
        bbox.y -= 3
        bbox.width += 6
        bbox.height += 3
        return bbox
    else:
        return None


def within_tolerance(x: int, expected_x: int, tol: float) -> bool:
    """
    Check if dimension is within tolerance of expected value.

    Parameters
    ----------
    x : int
        Actual dimension.
    expected_x : int
        Expected dimension.
    tol : float, optional
        Tolerance as fraction (default 0.1 = 10% is hardcoded in usage).

    Returns
    -------
    bool
        True if within tolerance.
    """
    return (1 - tol) <= (x / expected_x) <= (1 + tol)


def ms_threshold(image: MatLike, configuration: ThresholdConfiguration) -> MatLike:
    """
    Apply Otsu's thresholding and morphological closing.

    Parameters
    ----------
    image : MatLike
        Input image (grayscale or color).
    configuration : ThresholdConfiguration
        Threshold configuration.

    Returns
    -------
    MatLike
        Thresholded image.
    """
    if image.ndim > 2:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    kernel = np.ones(configuration.kernel_shape, np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    return thresh
