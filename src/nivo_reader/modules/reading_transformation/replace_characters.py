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

from typing import Any, final, override

import polars as pl

from .base import ReadingTransformation


@final
class ReplaceCharacters(ReadingTransformation):
    def __init__(self, column: str, mapping: dict[str, str], *conditions: pl.Expr):
        self.column = column
        self.mapping = mapping
        self.conditions = conditions

    @override
    def __call__(self, df: pl.DataFrame, *args: Any, **kwds: Any) -> pl.DataFrame:
        for what, with_ in self.mapping.items():
            df = df.with_columns(
                pl.when(pl.Expr.and_(*self.conditions))
                .then(pl.col(self.column).str.replace_all(what, with_, literal=True))
                .otherwise(pl.col(self.column))
                .alias(self.column)
            )
        return df
