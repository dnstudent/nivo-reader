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

import cv2
from cv2.typing import MatLike
from fancy_dataclass import JSONDataclass
from img2table.document.base.rotation import (
    get_connected_components,
    get_relevant_angles,
    estimate_skew,
)


from .base import Preprocessing

logger = logging.getLogger("nivo_reader.preprocessing.automatic_rotation")


@final
@dataclass
class AutomaticRotation(Preprocessing, JSONDataclass):
    name: str = "automatic_rotation"

    # Code partly taken from img2table. Credit goes to the authors
    @override
    def __call__(
        self, image: MatLike, previous_work: dict[str, Any] | None = None
    ) -> tuple[MatLike, dict[str, Any]]:
        if image.ndim == 2:
            gray = image.copy()
        elif previous_work and "_gray_image" in previous_work:
            gray = previous_work["_gray_image"]
        else:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

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

        # Compute affine transform. Takes angle in degrees.
        affine_transform = cv2.getRotationMatrix2D(
            center=(image.shape[1] / 2, image.shape[0] / 2),
            angle=skew_angle_degree,
            scale=1.0,
        )
        return affine_transform, {
            "rotated": True,
            "rotation_angle": skew_angle_degree,
        }
