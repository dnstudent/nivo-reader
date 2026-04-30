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
from typing import final, override, Any

from fancy_dataclass import JSONDataclass
import cv2
from cv2.typing import MatLike, Rect
import numpy as np

from nivo_reader.configuration.preprocessing import ThresholdConfiguration
from nivo_reader.image_processing import my_table_struct

from .base import TableDetection


@final
@dataclass
class NivoTableDetection(TableDetection, JSONDataclass):
    expected_table_shape: tuple[int, int]
    threshold_configuration: ThresholdConfiguration
    from_extracted_structure: bool

    @override
    def __call__(
        self, image: MatLike, previous_work: dict[str, Any] | None = None
    ) -> tuple[list[Rect] | None, dict[str, Any]]:
        if previous_work and "_gray_image" in previous_work:
            gray_image = previous_work["_gray_image"]
        else:
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        rect = try_detect_table_rect(
            gray_image, self.expected_table_shape, self.threshold_configuration, self.from_extracted_structure
        )
        if rect is None:
            return None, previous_work or {}
        return [rect], previous_work or {}
    
@final
@dataclass
class ParmaNivoTableDetection(TableDetection, JSONDataclass):
    @staticmethod
    def bbox2rect(bbox: Any) -> Rect:
        return (bbox.x1, bbox.y1, bbox.x2 - bbox.x1, bbox.y2 - bbox.y1)
    
    @override
    def __call__(self, image: MatLike, previous_work: dict[str, Any] | None = None) -> tuple[list[Rect] | None, dict[str, Any]]:
        from img2table.document import Image
        img = Image(cv2.imencode(".jpg", image)[1].tobytes(), detect_rotation=False)
        tables = img.extract_tables(implicit_rows=True)
        if len(tables) > 0:
            return list(map(lambda t: self.bbox2rect(t.bbox), tables)), previous_work or {}
        else:
            return None, previous_work or {}
        


def try_detect_table_rect(
    gray_image: MatLike,
    expected_table_shape: tuple[int, int],
    threshold_configuration: ThresholdConfiguration,
    from_extracted_structure: bool,
) -> Rect | None:
    """
    Detect table rectangle in image.

    Parameters
    ----------
    gray_image : MatLike
        Grayscale image.
    expected_table_shape : tuple[int, int]
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
    expected_table_width, expected_table_height = expected_table_shape
    bboxes = sorted(
        filter(
            lambda r: (
                within_tolerance(r[2], expected_table_width, tol=0.1)
                and within_tolerance(r[3], expected_table_height, tol=0.1)
            ),
            map(
                lambda cnt: cv2.boundingRect(cnt),
                cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0],
            ),
        ),
        key=lambda r: abs(expected_table_width * expected_table_height - r[2] * r[3]),
    )

    if len(bboxes) > 0:
        return bboxes[0]
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
