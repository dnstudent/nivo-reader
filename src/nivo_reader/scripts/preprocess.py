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
along with this program.  If not, see <https://www.gnu.org/licenses/>."""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
from cv2.typing import MatLike
from pydantic import (
    BaseModel,
    DirectoryPath,
    FilePath,
    Field,
)
from tqdm import tqdm

from nivo_reader.scripts.utils.paths import build_config_stack, mkopath
from nivo_reader.lib.images import read_matlike_image
from nivo_reader.modules.preprocessing.automatic_rotation import AutomaticRotation


class PipelineConfig(BaseModel):
    pass


class AppConfig(BaseModel):
    output_dir: Path
    project_dir: DirectoryPath
    debug_dir: Path | None = None
    image_formats: set[str] = {"png", "jpg", "jpeg", "gif"}
    overwrite: bool = False
    pipeline_config_fname: str = "config.toml"
    input_path: FilePath | DirectoryPath = Field(
        default_factory=lambda data: data["project_dir"]
    )
    logging_level: int


def preprocess(
    scan: MatLike,
    scan_config: PipelineConfig,
    debug_dir: Path | None,
) -> np.ndarray:
    rotator = AutomaticRotation()
    affine_transform, info = rotator(scan)
    return affine_transform


def setup_environment(args: argparse.Namespace):
    cli_params = {
        k: v
        for k, v in vars(args).items()
        if v is not None and k in AppConfig.model_fields
    }
    if "input_path" in cli_params and not Path(cli_params["input_path"]).is_absolute():
        cli_params["input_path"] = cli_params["project_dir"] / cli_params["input_path"]
    script_config = AppConfig(**cli_params)

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
            filename=Path(debug_dir) / "preprocess.log",
            filemode="a",
            format="[%(asctime)s][%(levelname)s]%(name)s - %(message)s",
        )
    else:
        logging.basicConfig(level=script_config.logging_level)

    return script_config


def main():
    parser = create_argparser()
    args = parser.parse_args()
    script_config = setup_environment(args)

    def to_preprocess(path: Path) -> bool:
        return (
            path.is_file()
            and (path.suffix.strip(".") in script_config.image_formats)
            and (
                script_config.overwrite
                or not mkopath(
                    path,
                    script_config.output_dir,
                    new_suffix=".npy",
                ).exists()
            )
        )

    iostack = map(
        lambda entry: (
            entry[0],
            entry[1],
            mkopath(entry[0], script_config.output_dir, new_suffix=".npy"),
        ),
        build_config_stack(
            root=script_config.project_dir,
            start=script_config.input_path,
            model=PipelineConfig,
            config_filename=script_config.pipeline_config_fname,
        ),
    )

    scan_config_stack = sorted(
        # Keep only entries that should be digitized (according to file format and overwrite rules)
        filter(
            lambda entry: to_preprocess(entry[0]),
            # Build the configuration tree
            iostack,
        ),
        key=lambda x: x[2],
    )

    pbar = tqdm(scan_config_stack, desc="Processing scans")
    for scan_path, scan_config, scan_output in pbar:
        logging.debug(
            f"Reading from {scan_path} to {scan_output} with conf {scan_config}"
        )
        if scan_config is None:
            pbar.write(
                f"✗ Error processing {scan_path.relative_to(script_config.project_dir)}: the configuration is invalid. Check the log."
            )
            continue
        pbar.set_description(f"Processing {scan_path.name}")
        scan_debug_dir = (
            mkopath(scan_path, Path(script_config.debug_dir), mkdir=True)
            if script_config.debug_dir
            else None
        )
        try:
            scan = read_matlike_image(scan_path)
            affine_transform = preprocess(scan, scan_config, scan_debug_dir)
            scan_output.parent.mkdir(parents=True, exist_ok=True)
            np.save(scan_output, affine_transform)
            pbar.write(
                f"✓ Processed: {scan_path.relative_to(script_config.project_dir)}"
            )
        except Exception as e:
            pbar.write(
                f"✗ Error processing {scan_path.relative_to(script_config.project_dir)}: {e}"
            )
            logging.exception(f"Error processing {scan_path}: {e}")


def create_argparser() -> argparse.ArgumentParser:
    """Create and configure argument parser for batch processing."""
    parser = argparse.ArgumentParser(
        prog="nivo-reader-preprocess",
        description="""Batch preprocessing of NIVO table images.""",
    )

    # Input/Output arguments
    _ = parser.add_argument(
        "-i",
        "--input-path",
        required=False,
        type=Path,
        help="Subset of the images to process. Could be a directory or a single image. Default is the project root.",
    )
    _ = parser.add_argument(
        "-o",
        "--output-dir",
        required=True,
        type=Path,
        help="Output directory for preprocessed files",
    )
    _ = parser.add_argument(
        "-d",
        "--debug-dir",
        type=Path,
        required=False,
        help="Base directory for debug artifacts. Optional.",
    )
    _ = parser.add_argument(
        "-p",
        "--project-dir",
        type=Path,
        required=True,
        help="The main directory of the project containing the NIVO table images in a subdirectory.",
    )
    # Overwrite flag
    _ = parser.add_argument(
        "-w",
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files",
    )

    # Image formats
    _ = parser.add_argument(
        "--image-formats",
        type=lambda s: set(s.split(",")),
        help=f"Comma-separated list of image file formats to process (default: {','.join(AppConfig.model_fields['image_formats'].default)}",
    )

    _ = parser.add_argument("--logging-level", type=int, default=logging.INFO)
    return parser


if __name__ == "__main__":
    main()
