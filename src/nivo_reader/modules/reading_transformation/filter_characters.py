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

from typing import final

import polars as pl

from nivo_reader.modules.reading_transformation.custom_substitution import (
    CustomSubstitution,
)


@final
class FilterCharacters(CustomSubstitution):
    def __init__(self, allowlist: str, *where: pl.Expr):
        super().__init__(
            "content",
            pl.col("content").str.replace_all(r"[^" + allowlist + "]", ""),
            *where,
        )
