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
from datetime import datetime
import hashlib
import logging
import sys
from pathlib import Path
from typing import Any
import cv2

from cv2.typing import MatLike
from pydantic import (
    BaseModel,
    DirectoryPath,
    FilePath,
)
from sqlalchemy import JSON, String, create_engine, DateTime, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker
from tqdm import tqdm

# from nivo_reader.scripts.utils.paths import mkopath
from nivo_reader.lib.images import read_matlike_image
from nivo_reader.modules.preprocessing.base import Preprocessor
from nivo_reader.modules.preprocessing.automatic_rotation import Img2TableRotation


STEP_NUMBER = "01"
STEP_NAME = "preprocess"


class Base(DeclarativeBase):
    pass


def get_preprocessing_model(table_name: str):
    class PreprocessingResult(Base):
        __tablename__: str = table_name
        __table_args__: dict[str, Any] = {"extend_existing": True}
        scanSHA256: Mapped[str] = mapped_column(String, nullable=False)
        runTag: Mapped[str] = mapped_column(String, primary_key=True)
        scanName: Mapped[str] = mapped_column(String, primary_key=True)
        infos: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
        created_at: Mapped[datetime] = mapped_column(
            DateTime,
            server_default=func.now(),  # Set once on insert
            nullable=False,
        )
        updated_at: Mapped[datetime] = mapped_column(
            DateTime,
            server_default=func.now(),
            onupdate=func.now(),  # Update on each change
            nullable=False,
        )

    return PreprocessingResult


class PipelineConfig(BaseModel):
    pass


class AppConfig(BaseModel):
    db_uri: str
    output_dir: Path
    table_name: str = "preprocess"
    run_tag: str
    project_dir: DirectoryPath
    debug_dir: Path | None = None
    image_formats: set[str] = {"png", "jpg", "jpeg", "gif"}
    overwrite: bool = False
    pipeline_config_fname: str = "config.toml"
    input_path: FilePath | DirectoryPath
    logging_level: int


def get_sha256(filepath: Path) -> str:
    sha256_hash = hashlib.sha256(usedforsecurity=False)
    with open(filepath, "rb") as f:
        for byte_block in iter(lambda: f.read(8192), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def preprocess(
    scan: MatLike,
    preprocessor: Preprocessor,
    scan_config: PipelineConfig,
    _debug_dir: Path | None = None,
) -> tuple[MatLike, dict[str, Any]]:
    preprocessed_image, infos = preprocessor(
        scan, scan_config.model_dump(mode="python")
    )
    return preprocessed_image, infos


def setup_environment(args: argparse.Namespace):
    cli_params = {
        k: v
        for k, v in vars(args).items()
        if v is not None and k in AppConfig.model_fields
    }

    project_dir = Path(cli_params.get("project_dir", args.project_dir))

    cli_params["db_uri"] = f"sqlite:///{project_dir}/db.sqlite"
    cli_params["output_dir"] = project_dir / f"{STEP_NUMBER}_{STEP_NAME}" / args.run_tag
    cli_params["output_dir"].mkdir(parents=True, exist_ok=True)

    if getattr(args, "debug", False):
        cli_params["debug_dir"] = (
            project_dir / "xx_debug" / f"{STEP_NUMBER}_{STEP_NAME}" / args.run_tag
        )

    if "input_path" not in cli_params:
        cli_params["input_path"] = project_dir / "00_input"
    elif not Path(cli_params["input_path"]).is_absolute():
        cli_params["input_path"] = project_dir / cli_params["input_path"]

    script_config = AppConfig(**cli_params)

    # Validate images directory
    if not script_config.input_path.is_file() and not script_config.input_path.is_dir():
        logging.error(f"Error: Input path {script_config.input_path} not valid")
        sys.exit(1)

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

    engine = create_engine(script_config.db_uri)
    PreprocessingResult = get_preprocessing_model(script_config.table_name)
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)

    def is_valid_file(path: Path) -> bool:
        return path.is_file() and (
            path.suffix.strip(".") in script_config.image_formats
        )

    iostack = (
        (path, PipelineConfig()) for path in script_config.input_path.rglob("**/*")
    )

    scan_config_stack = sorted(
        filter(
            lambda entry: is_valid_file(entry[0]),
            iostack,
        ),
        key=lambda x: x[0],
    )

    preprocessor = Img2TableRotation()

    pbar = tqdm(scan_config_stack, desc="Processing scans")
    with Session() as session:
        for scan_path, scan_config in pbar:
            logging.debug(f"Reading from {scan_path} with conf {scan_config}")
            pbar.set_description(f"Processing {scan_path.name}")

            try:
                scan_sha256 = get_sha256(scan_path)

                output_filename = scan_path.name.replace(scan_path.suffix, ".png")
                # Trick to get the file that may be inside a subdir
                output_file = (
                    list(script_config.output_dir.rglob(f"**/{output_filename}"))
                    + [script_config.output_dir / output_filename]
                )[0]

                db_entry = (
                    session.query(PreprocessingResult)
                    .filter_by(runTag=script_config.run_tag, scanName=scan_path.name)
                    .first()
                )
                fs_exists = output_file.exists()

                # Inconsistency check
                is_consistent = (db_entry is not None) == fs_exists
                if is_consistent and db_entry and fs_exists:
                    # Check if the scan itself has changed
                    if db_entry.scanSHA256 != scan_sha256:
                        is_consistent = False

                if (
                    not script_config.overwrite
                    and is_consistent
                    and db_entry
                    and fs_exists
                ):
                    pbar.write(
                        f"⏭ Skipped: {scan_path.relative_to(script_config.project_dir)} (already exists and consistent)"
                    )
                    continue

                scan = read_matlike_image(scan_path, grayscale=False)
                preprocessed_scan, infos = preprocess(scan, preprocessor, scan_config)

                # Save preprocessed scan (lossless PNG)
                _ = cv2.imwrite(str(output_file), preprocessed_scan)

                result = PreprocessingResult(
                    scanSHA256=scan_sha256,
                    runTag=script_config.run_tag,
                    scanName=scan_path.name,
                    infos=infos,
                )
                _ = session.merge(result)
                session.commit()

                pbar.write(
                    f"✓ Processed: {scan_path.relative_to(script_config.project_dir)}"
                )
            except Exception as e:
                session.rollback()
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
        help="Subset of the images to process. Could be a directory or a single image. Default is the project root's 00_input.",
    )
    _ = parser.add_argument(
        "--table-name",
        type=str,
        default="preprocess",
        help="Name of the table to store the results.",
    )
    _ = parser.add_argument(
        "--run-tag",
        required=True,
        type=str,
        help="Tag assigned by the user to the preprocessing run.",
    )
    _ = parser.add_argument(
        "-d",
        "--debug",
        action="store_true",
        help="Generate debug artifacts.",
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
        help="Overwrite existing output values in the DB",
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
