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

from typing import override, final

import polars as pl
import polars_distance  # pyright: ignore[reportUnusedImport]  # noqa: F401
from fancy_dataclass import JSONDataclass

from nivo_reader.modules.reading_transformation.base import ReadingTransformation


@final
class EditConfidence(ReadingTransformation, JSONDataclass):
    set_to: float
    conditions: tuple[pl.Expr, ...]

    def __init__(self, set_to: float, *conditions: pl.Expr):
        super().__init__()
        self.set_to = set_to
        self.conditions = conditions

    @override
    def __call__(self, df: pl.DataFrame) -> pl.DataFrame:
        return df.with_columns(
            pl.when(pl.Expr.and_(*self.conditions))
            .then(pl.lit(self.set_to))
            .otherwise(pl.col("confidence"))
            .alias("confidence"),
        )
