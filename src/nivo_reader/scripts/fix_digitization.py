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

from nivo_reader.modules.reading_transformation import (
    AssociateClosestMatch,
    NoOp,
    OverwriteCellContent,
    ReadingTransformationPipeline,
)
from nivo_reader.modules.reading_transformation.edit_confidence import EditConfidence
from nivo_reader.modules.reading_transformation.overwrite_cell_content import (
    OverwriteAndDropConfidence,
)
from nivo_reader.modules.reading_transformation.replace_characters import (
    ReplaceCharacters,
    ReplaceRegex,
)

from .utils.paths import mkopath

CONTENT_COLUMNS = pl.col("column") > 1
NUMERIC_COLUMNS = pl.col("column") != 0
PADDLEOCR = pl.col("reader") == "paddleocr"
EASYOCR = pl.col("reader") == "easyocr"


class PipelineConfig(BaseModel):
    confidence_threshold: float


class AppConfig(BaseModel):
    project_dir: DirectoryPath
    input_path: FilePath | DirectoryPath = Field(
        default_factory=lambda data: data["project_dir"]
    )
    output_dir: Path
    logging_level: int
    debug_dir: Path | None = None
    registry_file: FilePath | None = None


def fix_digitization(
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


def fix_digitizations_batch(
    input_dir: Path,
    output_dir: Path,
    pipeline_config: PipelineConfig,
    station_names: list[str] | None,
    _debug_dir: Path | None = None,
) -> None:
    """Process multiple raw NIVO digitizations and save cleaned versions.

    Args:
        input_dir: Directory containing raw digitization files
        output_dir: Directory where cleaned files will be saved
        station_names: List of valid station names for closest match resolution
        confidence_threshold: Confidence threshold for automatic easyocr corrections
    """
    output_dir.mkdir(exist_ok=True, parents=True)

    # Define the NIVO cleanup pipeline
    if station_names is not None:
        station_name_replacements = AssociateClosestMatch(
            pl.DataFrame({"column": 0, "content": station_names})
        )
    else:
        station_name_replacements = NoOp()

    dash_sub = OverwriteCellContent(
        "content",
        pl.lit("-"),
        (pl.col("content").str.contains_any(["-", "_", "=", "—", "−", "*", "→"]))
        | (
            EASYOCR
            & (pl.col("confidence") < pipeline_config.confidence_threshold)
            & (pl.col("content") == "2")
        )
        | pl.col("content").is_null()
        | (pl.col("content").str.strip_chars() == "")
        | (PADDLEOCR & pl.col("content").str.contains(r".", literal=True)),
        CONTENT_COLUMNS,
    )

    remove_dotdotdot = ReplaceRegex(
        "content", r"(\s*\.+\s*){2,}", "", pl.col("column") == 0
    )
    remove_unwanted_chars = ReplaceRegex(
        "content", r"[^\w\.\s']", "", pl.col("column") == 0
    )

    remove_content = OverwriteAndDropConfidence(
        "content",
        pl.lit(""),
        PADDLEOCR
        & NUMERIC_COLUMNS
        & (pl.col("content").str.len_chars() > 3)
        & (pl.col("content").str.contains(r"[a-zA-Z]")),
    )

    drop_value_chars = ReplaceCharacters(
        "content",
        {
            ":": "",
            "|": "",
            "i": "1",
            "l": "1",
            "I": "1",
            "C": "0",
            "»": ">",
            "•": "",
            ".": "",
            "'": "",
        },
        NUMERIC_COLUMNS,
    )
    drop_elev_chars = ReplaceCharacters("content", {"-": ""}, pl.col("column") == 1)
    drop_paddle_confidence = EditConfidence(
        0.0,
        PADDLEOCR & NUMERIC_COLUMNS & pl.col("content").str.contains(r"[^\d\-?>»\s]"),
    )

    pipeline = ReadingTransformationPipeline(
        remove_dotdotdot,
        remove_unwanted_chars,
        station_name_replacements,
        dash_sub,
        remove_content,
        drop_value_chars,
        drop_elev_chars,
        drop_paddle_confidence,
    )

    # Discover and process files
    raw_files = sorted(input_dir.rglob("*.arrow"))

    print(f"Fixing {len(raw_files)} files...")

    for input_path in tqdm(raw_files, desc="Fixing files"):
        output_path = mkopath(input_path, output_dir)
        raw_df = pl.read_ipc(input_path, memory_map=False)
        cleaned_df = fix_digitization(raw_df, pipeline)
        cleaned_df.write_ipc(output_path, compression="zstd")


def main():
    parser = create_argparser()
    args = parser.parse_args()
    script_config, pipeline_config = setup_environment(args)

    # Load station names for AssociateClosestMatch
    station_names = (
        (
            pl.read_excel(
                script_config.registry_file, has_header=True, columns=["Stazione"]
            )
            .filter(pl.col("Stazione").str.strip_chars() != "")
            .with_columns(pl.col("Stazione").str.split("/"))
            .explode("Stazione")["Stazione"]
            .drop_nulls()
            .to_list()
        )
        if script_config.registry_file is not None
        else None
    )

    fix_digitizations_batch(
        script_config.input_path,
        script_config.output_dir,
        pipeline_config,
        station_names,
        script_config.debug_dir,
    )


def setup_environment(args: argparse.Namespace):
    cli_params = {
        k: v
        for k, v in vars(args).items()
        if v is not None and k in AppConfig.model_fields
    }
    if "input_path" in cli_params and not Path(cli_params["input_path"]).is_absolute():
        cli_params["input_path"] = cli_params["project_dir"] / cli_params["input_path"]
    script_config = AppConfig(**cli_params)
    pipeline_config = PipelineConfig(confidence_threshold=args.confidence_threshold)

    # Validate images directory
    if not script_config.input_path.is_file() and not script_config.input_path.is_dir():
        logging.error(f"Error: Input path {script_config.input_path} not valid")
        sys.exit(1)

    output_dir = script_config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    debug_dir = script_config.debug_dir
    if debug_dir:
        Path(debug_dir).mkdir(exist_ok=True, parents=True)
        logging.basicConfig(
            level=script_config.logging_level,
            filename=Path(debug_dir) / "reader.log",
            filemode="w",
            format="[%(asctime)s][%(levelname)s]%(name)s - %(message)s",
        )
    else:
        logging.basicConfig(level=script_config.logging_level)

    return script_config, pipeline_config


def create_argparser():
    parser = argparse.ArgumentParser(
        prog="nivo-fix",
        description="Batch fix of raw NIVO table digitizations",
    )
    _ = parser.add_argument(
        "-i",
        "--input-path",
        help="Directory containing a subset of the raw NIVO table digitization files, or a single file.",
        type=Path,
        required=False,
    )
    _ = parser.add_argument(
        "-o",
        "--output-dir",
        help="Directory where the cleaned digitization files will be saved",
        type=Path,
        default=None,
    )
    _ = parser.add_argument(
        "-d",
        "--debug-dir",
        help="Directory where debug artefacts will be saved",
        type=Path,
        required=False,
    )
    _ = parser.add_argument(
        "-p",
        "--project-dir",
        type=Path,
        required=True,
        help="The main directory of the project containing the raw digitizations in a subdirectory.",
    )
    _ = parser.add_argument(
        "-r",
        "--registry-file",
        required=False,
        type=Path,
        help="Excel file with station names (column 'Stazione')",
    )
    _ = parser.add_argument(
        "-l",
        "--logging-level",
        type=int,
        default=logging.INFO,
        help="Logging level",
    )
    _ = parser.add_argument(
        "-c",
        "--confidence-threshold",
        type=float,
        default=0.8,
        help="Confidence threshold for automatic easyocr corrections",
    )
    return parser


if __name__ == "__main__":
    main()
