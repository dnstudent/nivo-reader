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
from .base import CellsDetection

from cv2.typing import MatLike, Rect


@final
@dataclass
class NivoCellsDetection(CellsDetection, JSONDataclass):
    @override
    def __call__(
        self, image: MatLike, table_rect: Rect
    ) -> tuple[list[Rect], list[tuple[int, int]], dict[str, Any]]:

        pass
