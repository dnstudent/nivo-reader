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
from typing import override

import polars as pl
import polars_distance  # pyright: ignore[reportUnusedImport]  # noqa: F401
from fancy_dataclass import JSONDataclass

from nivo_reader.modules.reading_transformation.base import ReadingTransformation


@dataclass
class AssociateClosestMatch(ReadingTransformation, JSONDataclass):
    options: pl.DataFrame

    @override
    def __call__(self, df: pl.DataFrame) -> pl.DataFrame:
        options = self.options.with_columns(
            simplified=pl.col("content")
            .str.to_lowercase()
            .str.strip_chars()
            .str.replace_all(r"\s+", " ")
        )
        df = df.with_columns(pl.row_index())
        matches_df = (
            df.with_columns(
                simplified=pl.col("content")
                .str.to_lowercase()
                .str.strip_chars()
                .str.replace_all(r"\s+", " ")
            )
            .join(
                options,
                how="inner",
                on=options.drop("content", "simplified").columns,
                suffix="_reference",
            )
            .with_columns(
                distance=(
                    pl.col("simplified")  # pyright: ignore[reportUnknownMemberType]
                    .dist_str.levenshtein("simplified_reference")  # pyright: ignore[reportAttributeAccessIssue]
                    .cast(pl.Int32)
                )
            )
            .sort("index", "distance")
            .group_by("index", maintain_order=True)
            .first(ignore_nulls=True)
            .with_columns(similarity=-pl.col("distance"))
            .select("index", "content_reference")
        )
        return (
            df.join(matches_df, on="index", how="left", validate="1:1")
            .with_columns(
                content=pl.coalesce(pl.col("content_reference"), pl.col("content"))
            )
            .drop("index", "content_reference")
        )
