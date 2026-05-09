"""
nivo-reader: a tool to digitize snowfall data tables from the Italian Hydrological Service
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
from collections.abc import Mapping

from cv2.typing import MatLike
from fancy_dataclass import JSONBaseDataclass, JSONDataclass

from nivo_reader.lib.common import BoundingBox


@dataclass
class CellSpec(JSONDataclass):
    row: int
    column: int
    cell_region: BoundingBox | None


@dataclass
class SpanSpec(JSONDataclass):
    start: int
    end: int


@dataclass
class HeaderSpec(JSONDataclass):
    header_region: BoundingBox


@dataclass
class IndexSpec(JSONDataclass):
    index_region: BoundingBox


@dataclass
class ContentSpec(JSONDataclass):
    content_region: BoundingBox
    # Flat list of cell specs
    cells: list[CellSpec]
    nrows: int
    ncols: int


@dataclass
class TableSpec(JSONDataclass):
    full_region: BoundingBox
    content_spec: ContentSpec
    header_spec: HeaderSpec | None
    index_spec: IndexSpec | None

    @property
    def nrows(self) -> int:
        return self.content_spec.nrows

    @property
    def ncols(self) -> int:
        return self.content_spec.ncols


@dataclass
class StructureResult(JSONDataclass):
    tables: list[TableSpec]


@dataclass
class StructureDetector(JSONBaseDataclass, ABC):
    name: str

    @abstractmethod
    def __call__(
        self,
        image: MatLike,
        configuration: Mapping[str, Any],
        previous_work: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> tuple[StructureResult, dict[str, Any]]:
        """
        Detects structure from a preprocessed image.

        Returns:
            A tuple where the first element is the structure results (containing full
            table boundary boxes and per-cell boundary boxes, with row/column indices),
            and the second element is the previous_work dictionary or additional infos.
        """
        pass
