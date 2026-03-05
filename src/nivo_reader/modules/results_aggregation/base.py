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

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import final, override

import polars as pl
from fancy_dataclass import JSONBaseDataclass, JSONDataclass


@dataclass
class ResultsAggregator(JSONBaseDataclass, ABC):
    @abstractmethod
    def __call__(self, df: pl.DataFrame) -> pl.DataFrame:
        """Aggregate the results from a DataFrame. Must always return a DataFrame where each "column" and "row" combination appears at most once."""
        raise NotImplementedError()


@final
@override
@dataclass
class AggregatorPipeline(ResultsAggregator, JSONDataclass):
    aggregators: list[ResultsAggregator] = field(
        default_factory=list, metadata={"doc": "List of aggregators to apply in order."}
    )

    @override
    def __call__(self, df: pl.DataFrame) -> pl.DataFrame:
        aggregates: list[pl.DataFrame] = []
        for aggregator in self.aggregators:
            aggregates.append(aggregator(df))
            df = df.join(aggregates[-1], on=["column", "row"], how="anti")
        return pl.concat(aggregates, how="vertical")
