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

import cv2
from cv2.typing import MatLike
from pydantic import BaseModel, DirectoryPath, FilePath, ValidationError
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session
from tqdm import tqdm

from nivo_reader.lib.images import read_matlike_image
from nivo_reader.models.db import (
    Base,
    CellStructure,
    PreprocessedScan,
    Project,
    StructureRun,
    TableStructure,
)
from nivo_reader.modules.structure_detection.base import (
    StructureDetector,
    StructureResult,
)
from nivo_reader.modules.structure_detection.nivo_structure_detection import (
    NivoStructureDetection,
)
from nivo_reader.modules.structure_detection.nivo_structure_detection import (
    NivoStructureDetectionConfig as PipelineConfig,
)
from nivo_reader.scripts.utils.paths import (
    build_config_dict,
    find_nested_files,
)

STEP_NUMBER = "02"
STEP_NAME = "table_structure"


class AppConfig(BaseModel):
    db_uri: str
    project_name: str
    preprocessing_run_tag: str
    structure_run_tag: str
    project_dir: DirectoryPath
    debug_dir: Path | None = None
    overwrite: bool = False
    input_path: FilePath | DirectoryPath
    logging_level: int


def draw_structure_debug(scan: MatLike, structure: StructureResult) -> MatLike:
    debug_image = scan.copy()
    for table in structure.tables:
        # Draw Content
        cr = table.content_spec.content_region
        _ = cv2.rectangle(
            debug_image,
            (cr.x, cr.y),
            (cr.x + cr.width, cr.y + cr.height),
            (0, 255, 0),
            2,
        )

        # Draw Header
        if table.header_spec:
            hr = table.header_spec.header_region
            _ = cv2.rectangle(
                debug_image,
                (hr.x, hr.y),
                (hr.x + hr.width, hr.y + hr.height),
                (255, 0, 0),
                2,
            )

        # Draw Index
        if table.index_spec:
            ir = table.index_spec.index_region
            _ = cv2.rectangle(
                debug_image,
                (ir.x, ir.y),
                (ir.x + ir.width, ir.y + ir.height),
                (0, 0, 255),
                2,
            )

        # Draw Cells
        for cell in table.content_spec.cells:
            if cell.cell_region:
                r = cell.cell_region
                _ = cv2.rectangle(
                    debug_image,
                    (r.x, r.y),
                    (r.x + r.width, r.y + r.height),
                    (230, 89, 173),
                    1,
                )

                _ = cv2.putText(
                    debug_image,
                    f"({cell.row},{cell.column})",
                    (r.x - 5, r.y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.3,
                    (230, 89, 173),
                    1,
                )

    return debug_image


def detect_structure(
    scan: MatLike,
    scan_config: PipelineConfig,
    detector: StructureDetector,
    debug_path: Path | None,
):
    structure, info = detector(scan, scan_config.model_dump(mode="python"))
    if debug_path and structure and structure.tables:
        scan_with_debug = draw_structure_debug(scan, structure)
        _ = cv2.imwrite(str(debug_path), scan_with_debug)
    return structure, info


def setup_environment(args: argparse.Namespace):
    cli_params = {
        k: v
        for k, v in vars(args).items()
        if v is not None and k in AppConfig.model_fields
    }

    project_dir = Path(cli_params.get("project_dir", args.project_dir))

    cli_params["db_uri"] = f"sqlite:///{project_dir}/db.sqlite"

    if getattr(args, "debug", False):
        cli_params["debug_dir"] = (
            project_dir
            / "xx_debug"
            / f"{STEP_NUMBER}_{STEP_NAME}"
            / args.structure_run_tag
        )

    if "input_path" not in cli_params:
        cli_params["input_path"] = (
            project_dir / "01_preprocess" / args.preprocessing_run_tag
        )
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
            filename=Path(debug_dir) / "structure_detection.log",
            filemode="a",
            format="[%(asctime)s][%(levelname)s]%(name)s - %(message)s",
        )
    else:
        logging.basicConfig(level=script_config.logging_level)

    return script_config


def get_or_init(
    project_name: str,
    preprocessing_run_tag: str,
    structure_run_tag: str,
    session: Session,
):
    project = session.scalars(
        select(Project).filter_by(name=project_name)
    ).one_or_none()
    if not project:
        raise ValueError(f"Project '{project_name}' not found")

    preprocessing_run = next(
        (r for r in project.preprocessing_runs if r.tag == preprocessing_run_tag),
        None,
    )
    if not preprocessing_run:
        raise ValueError(f"Preprocessing run '{preprocessing_run_tag}' not found")

    structure_run = next(
        (r for r in project.structure_runs if r.tag == structure_run_tag),
        None,
    )
    if not structure_run:
        structure_run = StructureRun(tag=structure_run_tag, project=project)
        session.add(structure_run)
        session.flush()

    # preprocessed_by_fname = {
    #     PreprocessedScan.filename(ps.scan): ps
    #     for ps in preprocessing_run.preprocessed_scans
    # }
    # structured_by_ps = {
    #     ts.preprocessedScan_id: ts for ts in structure_run.table_structures
    # }

    return preprocessing_run, structure_run


def get_scan_configs(
    input_dir: Path,
    preprocessed_map: dict[str, PreprocessedScan],
) -> dict[PreprocessedScan, PipelineConfig | ValidationError | KeyError]:
    scan_config_dict = build_config_dict(input_dir, PipelineConfig)
    return {
        preprocessed_map[scan_path.name]: config_or_err
        for scan_path, config_or_err in scan_config_dict.items()
        if scan_path.name in preprocessed_map
    }


def filter_invalid_configs(
    structure_configs: dict[
        PreprocessedScan, PipelineConfig | ValidationError | KeyError
    ],
) -> dict[PreprocessedScan, PipelineConfig]:
    valid: dict[PreprocessedScan, PipelineConfig] = {}
    for ps, config in structure_configs.items():
        if isinstance(config, ValidationError):
            logging.error(f"Validation error for {ps.scan.filename}: {config}")
        elif isinstance(config, KeyError):
            logging.error(f"Key error for {ps.scan.filename}: {config}")
        else:
            valid[ps] = config
    return valid


def todo(
    ps: PreprocessedScan,
    overwrite: bool,
    config: PipelineConfig,
    structured_by_ps: dict[int, TableStructure],
) -> bool:
    existing = structured_by_ps.get(ps.id)
    if overwrite or existing is None or existing.config != config.model_dump():
        return True
    return False


class NoTablesError(RuntimeError):
    def __init__(self, preprocessed_scan: PreprocessedScan):
        super().__init__(f"No tables detected in {preprocessed_scan.scan.filename}")


def detect_structure_and_persist(
    structure_run: StructureRun,
    preprocessed_scan: PreprocessedScan,
    config: PipelineConfig,
    scan_path: Path,
    detector: NivoStructureDetection,
    script_config: AppConfig,
    session: Session,
):
    scan = read_matlike_image(scan_path)
    structure, _ = detect_structure(
        scan,
        config,
        detector,
        script_config.debug_dir / scan_path.with_suffix(".jpg").name
        if script_config.debug_dir
        else None,
    )

    if len(structure.tables) == 0:
        raise NoTablesError(preprocessed_scan)

    for table in structure.tables:
        header_spec = table.header_spec
        content_spec = table.content_spec
        index_spec = table.index_spec

        table_result = TableStructure(
            run=structure_run,
            preprocessed_scan=preprocessed_scan,
            config=config.model_dump(),
            bbox=table.full_region.to_dict(),
            header=header_spec.to_dict() if header_spec else None,
            index=index_spec.to_dict() if index_spec else None,
            content=content_spec.to_dict(),
            nrows=content_spec.nrows,
            ncols=content_spec.ncols,
        )

        for cell in content_spec.cells:
            if cell.cell_region is None:
                continue
            table_result.cell_structures.append(
                CellStructure(
                    bbox=cell.cell_region.to_dict(),
                    config=config.model_dump(),
                    row=cell.row,
                    col=cell.column,
                )
            )
            # session.add(cell_result)

        session.add(table_result)
        session.flush()  # To get the table_result.id


def main():
    parser = create_argparser()
    args = parser.parse_args()
    script_config = setup_environment(args)

    engine = create_engine(script_config.db_uri)
    Base.metadata.create_all(engine)

    detector = NivoStructureDetection()

    with Session(engine) as session, session.begin():
        (
            preprocessing_run,
            structure_run,
            # preprocessed_by_fname,
            # structured_by_ps,
        ) = get_or_init(
            script_config.project_name,
            script_config.preprocessing_run_tag,
            script_config.structure_run_tag,
            session,
        )

        scan_map = {ps.filename: ps for ps in preprocessing_run.preprocessed_scans}

        _structure_configs = build_config_dict(
            script_config.input_path,
            PipelineConfig,
        )
        structure_configs = {
            scan_map[k.name]: v for k, v in _structure_configs.items() if k in scan_map
        }

        scan_paths = find_nested_files(scan_map, script_config.input_path)
        detected_scans = set(
            map(lambda x: x.preprocessed_scan, structure_run.table_structures)
        )

        n_processed = 0
        failed_scans: list[str] = []
        pbar = tqdm(preprocessing_run.preprocessed_scans, desc="Processing scans")
        for pscan in pbar:
            pbar.set_description(f"Processing {pscan.scan.filename}")

            scan_path = scan_paths.get(pscan)
            if not scan_path:
                pbar.write(f"Scan {pscan.scan.filename} not found")
                logging.error(f"Scan {pscan.scan.filename} not found")
                continue

            scan_config = structure_configs[pscan]
            if isinstance(scan_config, (ValidationError, KeyError, Exception)):
                pbar.write(f"Invalid config for {pscan.scan.filename}")
                logging.error(
                    f"Invalid config for {pscan.scan.filename}: {scan_config}"
                )
                continue
            # Drop stale entries if overwriting
            if pscan in detected_scans and script_config.overwrite:
                pscan.table_structures.clear()
                session.flush()

            try:
                detect_structure_and_persist(
                    structure_run=structure_run,
                    preprocessed_scan=pscan,
                    config=scan_config,
                    scan_path=scan_path,
                    detector=detector,
                    script_config=script_config,
                    session=session,
                )
            except NoTablesError as e:
                pbar.write(f"No tables detected in {scan_path.name}")
                failed_scans.append(scan_path.name)
                logging.exception(f"No tables detected in {scan_path}: {e}")
            else:
                n_processed += 1

                pbar.write(
                    f"✓ Processed: {scan_path.relative_to(script_config.project_dir)}"
                )

        pbar.write(
            f"\nProcessing complete: {n_processed} scans processed, {len(failed_scans)} failed."
        )


def create_argparser() -> argparse.ArgumentParser:
    """Create and configure argument parser for batch processing."""
    parser = argparse.ArgumentParser(
        prog="nivo-reader-structure-detection",
        description="""Batch structure detection of NIVO table images.""",
    )

    # # Input/Output arguments
    # _ = parser.add_argument(
    #     "-i",
    #     "--input-path",
    #     required=False,
    #     type=Path,
    #     help="Directory of preprocessed scans to process. Defaults to project_dir/01_preprocess/preprocessing_run_tag.",
    # )
    _ = parser.add_argument(
        "--project-name",
        type=str,
        required=True,
        help="The name of the project.",
    )
    _ = parser.add_argument(
        "--preprocessing-run-tag",
        required=True,
        type=str,
        help="Tag of the preprocessing run the scans come from.",
    )
    _ = parser.add_argument(
        "--structure-run-tag",
        required=True,
        type=str,
        help="Tag assigned by the user to the current structure detection run.",
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

    _ = parser.add_argument("--logging-level", type=int, default=logging.INFO)
    return parser


if __name__ == "__main__":
    main()
