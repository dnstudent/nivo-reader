#!/usr/bin/env python3
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

import argparse
from pathlib import Path

import pandas as pd
import polars as pl
from openpyxl import Workbook, load_workbook
from tqdm import tqdm

from nivo_reader.modules.results_aggregation import (
    AggregatorPipeline,
    HighestScoring,
    MostOccurringValues,
)
from nivo_reader.modules.results_aggregation.base import ResultsAggregator

from .utils.excel_styling import ExcelStyler, NivoDefaultStyler, apply_styler
from .utils.paths import discover_files, reroute_file


def create_parser():
    parser = argparse.ArgumentParser(
        prog="nivo-writer",
        description="Batch conversion of NIVO table digitizations to Excel",
    )
    _ = parser.add_argument(
        "input_dir",
        help="Directory containing the NIVO table digitization files",
        type=str,
    )
    _ = parser.add_argument(
        "--output-dir",
        help="Directory where the Excel files will be saved",
        type=str,
        default=None,
    )
    _ = parser.add_argument(
        "--header-template",
        help="Path to a template Excel file to use as a header",
        type=str,
        default=None,
    )
    _ = parser.add_argument(
        "--confidence-threshold",
        help="Confidence threshold for highlighting cells",
        type=float,
        default=None,
    )
    return parser


def write_excel(
    digitization_file: Path,
    output_path: Path,
    header_template: Path | None,
    aggregator: ResultsAggregator,
    styler: ExcelStyler,
):
    row = 0
    col = 0
    if header_template and header_template.exists():
        wb = load_workbook(header_template)
        ws = wb.active

        if ws:
            row = ws.max_row
            col = ws.min_column - 1

            ws.title = "Sheet1"
        else:
            wb.create_sheet("Sheet1")

        wb.save(output_path)
    else:
        wb = Workbook()
        wb.create_sheet("Sheet1")
        wb.save(output_path)

    wb.close()

    mode = "a" if header_template else "w"
    if_sheet_exists = "overlay" if header_template else None
    aggregated_df = aggregator(pl.read_json(digitization_file))
    with pd.ExcelWriter(
        output_path, engine="openpyxl", mode=mode, if_sheet_exists=if_sheet_exists
    ) as xlwriter:
        (
            aggregated_df.pivot(
                on="column", index="row", values="content", sort_columns=True
            )
            .sort(by="row")
            .drop("row")
            .to_pandas()
            .to_excel(  # pyright: ignore[reportUnknownMemberType]
                xlwriter,
                startrow=row,
                startcol=col,
                engine="openpyxl",
                index=False,
                header=False,
            )
        )

    # Calculate table region (1-indexed for openpyxl)
    num_rows = aggregated_df.get_column("row").n_unique()
    num_cols = aggregated_df.get_column("column").n_unique()
    table_region = (row + 1, col + 1, row + num_rows, col + num_cols)

    # Apply styling
    apply_styler(output_path, styler, aggregated_df, table_region)


def main():
    parser = create_parser()
    args = parser.parse_args()
    input_dir = Path(args.input_dir)
    assert input_dir.exists(), f"Input directory {input_dir} does not exist"
    if args.output_dir is None:
        output_dir = input_dir
    else:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(exist_ok=True)
    header_template = Path(args.header_template) if args.header_template else None

    digitization_files = discover_files(input_dir, "digitization.json")
    aggregator = AggregatorPipeline(
        [MostOccurringValues(at_least=2), HighestScoring()],
    )
    styler = NivoDefaultStyler(threshold=0.9)

    for digitization_file in tqdm(digitization_files):
        output_path = reroute_file(
            digitization_file, output_dir, input_dir
        ).with_suffix(".xlsx")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        write_excel(digitization_file, output_path, header_template, aggregator, styler)


if __name__ == "__main__":
    main()
