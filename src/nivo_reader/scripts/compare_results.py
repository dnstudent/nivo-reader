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

import re
from pathlib import Path

import polars as pl


def digitization_to_long_format(path: Path):
    id_re = r"(?P<department>\w+)_(?P<year>\d{4})_(?P<n>\d+)\.gif"
    id_specs = re.fullmatch(id_re, path.parent.name)
    if id_specs:
        id_specs = id_specs.groupdict()
    return pl.read_json(path).with_columns(table=pl.lit(id_specs)).unnest("table")
