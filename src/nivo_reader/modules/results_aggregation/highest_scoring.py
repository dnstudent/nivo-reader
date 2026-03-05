"""nivo-reader: a tool to automatically read meteorological data
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
along with this program.  If not, see <https://www.gnu.org/licenses/>."""

from dataclasses import dataclass
from typing import final, override

import polars as pl
from fancy_dataclass import JSONDataclass

from .base import ResultsAggregator


@final
@dataclass
class HighestScoring(ResultsAggregator, JSONDataclass):
    @override
    def __call__(self, df: pl.DataFrame) -> pl.DataFrame:
        return (
            df.sort("column", "row", "confidence")
            .group_by(["column", "row"], maintain_order=True)
            .last()
            .select(["column", "row", "content", "confidence"])
        )
