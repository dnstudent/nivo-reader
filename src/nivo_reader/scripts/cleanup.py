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

import polars as pl
from tqdm import tqdm

from nivo_reader.modules.reading_transformation import (
    AssociateClosestMatch,
    CustomSubstitution,
    ReadingTransformationPipeline,
)
from nivo_reader.modules.reading_transformation.filter_characters import (
    FilterCharacters,
)

from .utils.paths import discover_files, reroute_file


def create_parser():
    parser = argparse.ArgumentParser(
        prog="nivo-cleanup",
        description="Batch cleanup of raw NIVO table digitizations",
    )
    _ = parser.add_argument(
        "input_dir",
        help="Directory containing the raw NIVO table digitization files",
        type=str,
    )
    _ = parser.add_argument(
        "--output-dir",
        help="Directory where the cleaned digitization files will be saved",
        type=str,
        default=None,
    )
    _ = parser.add_argument(
        "--anagrafica-file",
        required=True,
        type=str,
        help="File with station names (Excel file with 'Stazione' column)",
    )
    _ = parser.add_argument(
        "--confidence-threshold",
        help="Confidence threshold for automatic easyocr corrections (default: 0.8)",
        type=float,
        default=0.8,
    )
    return parser


def cleanup_digitization(
    raw_df: pl.DataFrame,
    pipeline: ReadingTransformationPipeline,
) -> pl.DataFrame:
    """Apply cleanup pipeline to a single digitization dataframe.

    Args:
        raw_df: Raw digitization dataframe
        pipeline: The ReadingTransformationPipeline to apply

    Returns:
        Cleaned digitization dataframe
    """
    return pipeline(raw_df)


def cleanup_digitizations_batch(
    input_dir: Path,
    output_dir: Path,
    station_names: list[str],
    confidence_threshold: float = 0.8,
) -> None:
    """Process multiple raw NIVO digitizations and save cleaned versions.

    Args:
        input_dir: Directory containing raw digitization files
        output_dir: Directory where cleaned files will be saved
        station_names: List of valid station names for closest match resolution
        confidence_threshold: Confidence threshold for automatic easyocr corrections
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    # Define the NIVO cleanup pipeline
    station_name_replacements = AssociateClosestMatch(
        pl.DataFrame({"column": 0, "content": station_names})
    )
    dash_sub = CustomSubstitution(
        "content",
        pl.lit("-"),
        (pl.col("content").str.contains_any(["-", "_", "=", "—", "−"]))
        & (pl.col("column") > 1),
        (pl.col("reader") == "easyocr")
        & (pl.col("confidence") < confidence_threshold)
        & (pl.col("content") == "2")
        & (pl.col("column") > 1),
        (pl.col("content").is_null() | (pl.col("content") == ""))
        & (pl.col("column") > 1),
    )
    allowed_chars = FilterCharacters(
        "0123456789-",
        pl.col("column") != 0,
    )

    pipeline = ReadingTransformationPipeline(
        station_name_replacements,
        dash_sub,
        allowed_chars,
    )

    # Discover and process files
    raw_files = discover_files(input_dir, "raw_digitization.json")

    if not raw_files:
        print(f"No raw_digitization.json files found in {input_dir}")
        return

    print(f"Cleaning {len(raw_files)} files...")
    for raw_file in tqdm(raw_files, desc="Cleaning files"):
        output_path = reroute_file(raw_file, output_dir, input_dir).with_name(
            "digitization.json"
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)

        raw_df = pl.read_json(raw_file)
        cleaned_df = cleanup_digitization(raw_df, pipeline)
        cleaned_df.write_json(output_path)


def main():
    parser = create_parser()
    args = parser.parse_args()
    input_dir = Path(args.input_dir)
    assert input_dir.exists(), f"Input directory {input_dir} does not exist"

    if args.output_dir is None:
        output_dir = input_dir
    else:
        output_dir = Path(args.output_dir)

    # Load station names for AssociateClosestMatch
    station_names = (
        pl.read_excel(args.anagrafica_file)["Stazione"].drop_nulls().to_list()
    )

    cleanup_digitizations_batch(
        input_dir=input_dir,
        output_dir=output_dir,
        station_names=station_names,
        confidence_threshold=args.confidence_threshold,
    )


if __name__ == "__main__":
    main()
