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
from typing import Any, override

from fancy_dataclass import JSONBaseDataclass

from cv2.typing import MatLike


@dataclass
class Preprocessing(JSONBaseDataclass, ABC):
    name: str

    @abstractmethod
    def __call__(self, image: MatLike) -> tuple[MatLike, dict[str, Any]]:
        pass


@dataclass
class PreprocessingPipeline(Preprocessing):
    preprocessors: list[Preprocessing]

    @override
    def __call__(
        self, image: MatLike, previous_work: dict[str, Any] | None = None
    ) -> tuple[MatLike, dict[str, Any]]:
        infos = {}
        for preprocessor in self.preprocessors:
            image, info = preprocessor(image)
            infos[preprocessor.name] = info
        return image, infos
