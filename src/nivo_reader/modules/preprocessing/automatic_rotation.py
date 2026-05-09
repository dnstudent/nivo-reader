# TODO: credits to img2table authors
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
from typing import Any, final, override
import logging
from collections.abc import Mapping

import cv2
from cv2.typing import MatLike
from img2table.document.base.rotation import (
    get_connected_components,
    get_relevant_angles,
    estimate_skew,
    rotate_img_with_border,
)

from .base import Preprocessor

logger = logging.getLogger("nivo_reader.preprocessing.automatic_rotation")


@final
@dataclass
class Img2TableRotation(Preprocessor):
    name: str = "img2table_rotation"

    # Code partly taken from img2table. Credit goes to the authors
    @override
    def __call__(
        self,
        image: MatLike,
        configuration: Mapping[str, Any],
        previous_work: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> tuple[MatLike, dict[str, Any]]:
        """Automatically rotate the image so that its lines are straight.

        Args:
            image (MatLike): input image
            configuration (Mapping[str, Any]): configuration of the preprocessor. Ignored.
            previous_work (dict[str, Any] | None, optional): previous work. If provided, the gray image will be taken from previous_work["_gray_image"]. Defaults to None.
            **kwargs (Any): additional keyword arguments

        Returns:
            tuple[MatLike, dict[str, Any]]: rotated image and a dictionary of informations
        """
        if image.ndim == 2:
            gray = image
        elif previous_work and "_gray_image" in previous_work:
            gray = previous_work["_gray_image"]
        else:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        cc_centroids, ref_height, thresh = get_connected_components(img=gray)

        # Check number of centroids
        if len(cc_centroids) < 2:
            return image, {"rotated": False, "rotation_angle": 0.0}

        # Compute most likely angles (in degrees) from connected components
        angles_degree = get_relevant_angles(
            centroids=cc_centroids, ref_height=ref_height
        )
        # Estimate skew
        skew_angle_degree = estimate_skew(angles=angles_degree, thresh=thresh)

        return rotate_img_with_border(image, skew_angle_degree), {
            "rotated": True,
            "rotation_angle": skew_angle_degree,
        }
