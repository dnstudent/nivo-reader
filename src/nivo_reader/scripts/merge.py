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
import logging
import sys
from pathlib import Path

import polars as pl
from pydantic import BaseModel, DirectoryPath, Field, FilePath
from tqdm import tqdm

from nivo_reader.modules.results_aggregation import (
    AggregatorPipeline,
    HighestScoring,
    MostOccurringValues,
)
from nivo_reader.modules.results_aggregation.base import ResultsAggregator

from .utils.paths import mkopath


class AppConfig(BaseModel):
    project_dir: DirectoryPath
    input_path: FilePath | DirectoryPath = Field(
        default_factory=lambda data: data["project_dir"]
    )
    output_dir: Path
    logging_level: int


def create_argparser():
    parser = argparse.ArgumentParser(
        prog="nivo-merger",
        description="Merge results from multiple OCR readers into a single results table",
    )
    _ = parser.add_argument(
        "-i",
        "--input-path",
        help="Directory containing the NIVO table digitization files",
        type=Path,
        required=False,
    )
    _ = parser.add_argument(
        "-o",
        "--output-dir",
        help="Directory where the merged Arrow files will be saved",
        type=Path,
        required=True,
    )
    _ = parser.add_argument(
        "-p",
        "--project-dir",
        type=Path,
        required=True,
        help="The main directory of the project containing the digitization files in a subdirectory.",
    )
    _ = parser.add_argument(
        "-l",
        "--logging-level",
        type=int,
        default=logging.INFO,
        help="Logging level",
    )
    return parser


def merge_digitization_results(
    digitization_df: pl.DataFrame,
    aggregator: ResultsAggregator,
) -> pl.DataFrame:
    """Create a styled Excel Workbook from a single digitization dataframe.

    Args:
        digitization_df: The input digitization dataframe (raw or cleaned)
        aggregator: The ResultsAggregator pipeline to apply

    Returns:
        The merged and aggregated dataframe
    """
    return aggregator(digitization_df)


def merge_digitization_batch(
    input_dir: Path, output_dir: Path, aggregator: ResultsAggregator
) -> None:
    """Process multiple digitizations and save merged Arrow files.

    Args:
        input_dir: Directory containing digitization files
        output_dir: Directory where merged files will be saved
        aggregator: Aggregator logic to condense the OCR readings
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    digitization_files = sorted(input_dir.rglob("*.arrow"))

    for digitization_file in tqdm(digitization_files, desc="Writing merged files"):
        output_path = mkopath(digitization_file, output_dir)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        digitization_df = pl.read_ipc(digitization_file, memory_map=False)
        merged_df = merge_digitization_results(digitization_df, aggregator)
        merged_df.write_ipc(output_path, compression="zstd")


def setup_environment(args: argparse.Namespace) -> AppConfig:
    cli_params = {
        k: v
        for k, v in vars(args).items()
        if v is not None and k in AppConfig.model_fields
    }
    if "input_path" in cli_params and not Path(cli_params["input_path"]).is_absolute():
        cli_params["input_path"] = cli_params["project_dir"] / cli_params["input_path"]

    script_config = AppConfig(**cli_params)

    # Validate input path
    if not script_config.input_path.exists():
        logging.error(f"Error: Input path {script_config.input_path} not valid")
        sys.exit(1)

    output_dir = script_config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(level=script_config.logging_level)

    return script_config


def main():
    parser = create_argparser()
    args = parser.parse_args()
    script_config = setup_environment(args)

    aggregator = AggregatorPipeline(
        index_columns=["row", "column"],
        aggregators=[
            MostOccurringValues(index_columns=["row", "column"], at_least=2),
            HighestScoring(index_columns=["row", "column"]),
        ],
    )

    merge_digitization_batch(
        input_dir=(
            script_config.input_path
            if script_config.input_path.is_dir()
            else script_config.input_path.parent
        ),
        output_dir=script_config.output_dir,
        aggregator=aggregator,
    )


if __name__ == "__main__":
    main()
