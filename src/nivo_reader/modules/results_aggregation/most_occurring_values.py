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
along with this program.  If not, see <https://www.gnu.org/licenses/>."""

from dataclasses import dataclass, field
from typing import override

import polars as pl
from fancy_dataclass import JSONDataclass

from .base import ResultsAggregator


@dataclass
class MostOccurringValues(ResultsAggregator, JSONDataclass):
    at_least: int = field(
        metadata={"doc": "Minimum number of occurrences for a value to be considered."}
    )

    @override
    def __call__(self, df: pl.DataFrame) -> pl.DataFrame:
        most_occurring = (
            df.sort("column", "row", "content")
            .group_by("column", "row", maintain_order=True)
            .agg(pl.col("content").rle())
            .explode("content")
            .unnest("content")
            .filter(pl.col("len") >= self.at_least)
            .sort("column", "row", "len", descending=[False, False, True])
            .group_by("column", "row", maintain_order=True)
            .first()
            .select("column", "row", "value")
            .rename({"value": "content"})
        )
        highest_confidence = (
            df.select("column", "row", "content", "confidence")
            .group_by("column", "row", "content")
            .agg(pl.col("confidence").max())
        )
        return most_occurring.join(
            highest_confidence,
            on=["column", "row", "content"],
            how="left",
        )
