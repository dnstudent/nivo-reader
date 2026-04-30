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

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from cv2.typing import MatLike, Rect

from fancy_dataclass import JSONBaseDataclass


@dataclass
class TableDetection(JSONBaseDataclass, ABC):
    name: str

    @abstractmethod
    def __call__(
        self, image: MatLike, previous_work: dict[str, Any] | None = None
    ) -> tuple[list[Rect] | None, dict[str, Any]]:
        pass
