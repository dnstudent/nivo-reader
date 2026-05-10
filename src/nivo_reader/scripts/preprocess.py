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
from typing import Any, cast
import cv2

from cv2.typing import MatLike
from pydantic import (
    BaseModel,
    DirectoryPath,
    FilePath,
    ValidationError,
)
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session
from tqdm import tqdm

# from nivo_reader.scripts.utils.paths import mkopath
from nivo_reader.lib.images import read_matlike_image
from nivo_reader.modules.preprocessing.base import Preprocessor
from nivo_reader.modules.preprocessing.automatic_rotation import Img2TableRotation
from nivo_reader.models.db import (
    Base,
    PreprocessedScan,
    PreprocessingRun,
    Project,
    Scan,
)
from .utils.paths import build_config_dict, find_nested_files, get_sha256

STEP_NUMBER = "01"
STEP_NAME = "preprocess"


class PipelineConfig(BaseModel):
    pass


class AppConfig(BaseModel):
    db_uri: str
    run_tag: str
    output_dir: Path
    project_dir: DirectoryPath
    project_name: str
    debug_dir: Path | None = None
    overwrite: bool = False
    input_path: FilePath | DirectoryPath
    logging_level: int


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

    cli_params["input_path"] = project_dir / "00_input"

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


def get_or_init(run_tag: str, project_name: str, session: Session):
    project = session.execute(
        select(Project).filter_by(name=project_name)
    ).scalar_one_or_none()
    if not project:
        raise ValueError(f"Project '{project_name}' not found")
    run: PreprocessingRun | None = next(
        (r for r in project.preprocessing_runs if r.tag == run_tag),
        None,
    )
    if not run:
        run = PreprocessingRun(tag=run_tag, project=project, preprocessed_scans=[])
        session.add(run)
    preprocessed = run.preprocessed_scans
    return project, run, {p.scan: p for p in preprocessed}


def todo(
    scan: Scan,
    overwrite: bool,
    config: PipelineConfig,
    already_processed_map: dict[Scan, PreprocessedScan],
):
    if (
        overwrite
        or scan not in already_processed_map
        or already_processed_map[scan].config != config.model_dump()
    ):
        return True
    return False


def get_scan_configs(
    input_dir: Path, scans: set[Scan]
) -> dict[Scan, PipelineConfig | ValidationError | KeyError]:
    scan_config_dict = build_config_dict(input_dir, PipelineConfig)
    scans_by_fname = {scan.filename: scan for scan in scans}
    return {
        scans_by_fname[scan_path.name]: config_or_err
        for scan_path, config_or_err in scan_config_dict.items()
        if scan_path.name in scans_by_fname
    }


def filter_invalid_configs(
    preprocessing_configs: dict[Scan, PipelineConfig | ValidationError | KeyError],
) -> dict[Scan, PipelineConfig]:
    valid_configs: dict[Scan, PipelineConfig] = {}
    for scan, config in preprocessing_configs.items():
        if isinstance(config, ValidationError):
            logging.error(f"Validation error for {scan.filename}: {config}")
        elif isinstance(config, KeyError):
            logging.error(f"Key error for {scan.filename}: {config}")
        else:
            valid_configs[scan] = config
    return valid_configs


def main():
    parser = create_argparser()
    args = parser.parse_args()
    script_config = setup_environment(args)

    preprocessor = Img2TableRotation()

    engine = create_engine(script_config.db_uri)
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        project, run, preprocessed_map = get_or_init(
            script_config.run_tag, script_config.project_name, session
        )
        all_scans = set(project.scans)
        preprocessing_configs = filter_invalid_configs(
            get_scan_configs(script_config.input_path, all_scans)
        )
        todo_scans = sorted(
            [
                scan
                for scan in all_scans
                if todo(
                    scan,
                    script_config.overwrite,
                    preprocessing_configs[scan],
                    preprocessed_map,
                )
            ],
            key=lambda s: s.filename,
        )
        scans_paths = find_nested_files(
            {scan.filename: scan for scan in all_scans},
            script_config.input_path,
        )

        n_processed = 0
        n_failed = 0
        pbar = tqdm(todo_scans, desc="Processing scans")
        for scan in pbar:
            pbar.set_description(f"Processing {scan.filename}")

            output_path = PreprocessedScan.find_path(scan, script_config.output_dir)
            scan_path = scans_paths.get(scan)
            if not scan_path:
                pbar.write(f"Scan {scan.filename} not found")
                logging.error(f"Scan {scan.filename} not found")
                continue

            scan_config = preprocessing_configs[scan]
            try:
                scan_data = read_matlike_image(scan_path)
                preprocessed_scan, _ = preprocess(scan_data, preprocessor, scan_config)

                # Save preprocessed scan (lossless PNG)
                _ = cv2.imwrite(str(output_path), preprocessed_scan)

                result = preprocessed_map.get(scan)
                if not result:
                    result = PreprocessedScan(run=run, scan=scan)
                    session.add(result)
                    preprocessed_map[scan] = result

                result.sha256Hash = get_sha256(cast(Path, output_path))
                result.config = scan_config.model_dump()
                session.commit()
                n_processed += 1

                pbar.write(
                    f"✓ Processed: {scan_path.relative_to(script_config.project_dir)}"
                )
            except Exception as e:
                session.rollback()
                n_failed += 1
                pbar.write(
                    f"✗ Error processing {scan_path.relative_to(script_config.project_dir)}: {e}"
                )
                logging.exception(f"Error processing {scan_path}: {e}")

        pbar.write(
            f"\nProcessing complete: {n_processed} scans processed, {n_failed} failed."
        )


def create_argparser() -> argparse.ArgumentParser:
    """Create and configure argument parser for batch processing."""
    parser = argparse.ArgumentParser(
        prog="nivo-reader-preprocess",
        description="""Batch preprocessing of NIVO table images.""",
    )

    # Input/Output arguments
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
    _ = parser.add_argument(
        "--project-name",
        type=str,
        required=True,
        help="The name of the project.",
    )

    # Overwrite flag
    _ = parser.add_argument(
        "-w",
        "--overwrite",
        action="store_true",
        help="Overwrite existing output values in the DB",
    )

    _ = parser.add_argument("--logging-level", type=int, default=logging.INFO)
    return parser


if __name__ == "__main__":
    main()
