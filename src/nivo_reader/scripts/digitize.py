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
from datetime import datetime
from pathlib import Path
from string import ascii_letters
from typing import Any, cast

import easyocr
import polars as pl
import tesserocr
from pydantic import BaseModel, DirectoryPath, FilePath
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session
from tqdm import tqdm

from nivo_reader.lib.images import read_matlike_image
from nivo_reader.models.db import (
    Base,
    CellContent,
    DigitizationRun,
    Project,
    TableStructure,
)
from nivo_reader.modules.preprocessing.base import PreprocessingPipeline
from nivo_reader.modules.preprocessing.image_cleaning import Binarization, LineEraser
from nivo_reader.modules.table_digitization import CellsListDigitizer, MultipleOCRModule
from nivo_reader.modules.table_digitization.ocrs.easyocr import (
    EasyOCRParagraph,
    EasyOCRWord,
)
from nivo_reader.modules.table_digitization.ocrs.paddleocr import PaddleOCR
from nivo_reader.modules.table_digitization.ocrs.tesserocr import (
    build_tesserocr_number_ocr,
    build_tesserocr_text_ocr,
)
from nivo_reader.scripts.utils.paths import find_nested_files

STEP_NUMBER = "03"
STEP_NAME = "digitization"


class PipelineConfig(BaseModel):
    pass


class AppConfig(BaseModel):
    db_uri: str
    project_name: str
    preprocessing_run_tag: str
    structure_run_tag: str
    digitization_run_tag: str
    project_dir: DirectoryPath
    debug_dir: Path | None = None
    overwrite: bool = False
    input_path: FilePath | DirectoryPath
    scan_list: FilePath | None
    logging_level: int


def setup_environment(args: argparse.Namespace):
    cli_params = {k: v for k, v in vars(args).items() if k in AppConfig.model_fields}

    project_dir = Path(cli_params.get("project_dir", args.project_dir))

    cli_params["db_uri"] = f"sqlite+pysqlite:///{project_dir}/db.sqlite"

    if getattr(args, "debug", False):
        cli_params["debug_dir"] = (
            project_dir
            / "xx_debug"
            / f"{STEP_NUMBER}_{STEP_NAME}"
            / args.digitization_run_tag
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
            filename=Path(debug_dir) / "digitization.log",
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
    digitization_run_tag: str,
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
        raise ValueError(f"Structure run '{structure_run_tag}' not found")

    digitization_run = next(
        (r for r in project.digitization_runs if r.tag == digitization_run_tag),
        None,
    )
    if not digitization_run:
        digitization_run = DigitizationRun(tag=digitization_run_tag, project=project)
        session.add(digitization_run)
        session.flush()

    return preprocessing_run, structure_run, digitization_run


def digitize_and_persist(
    digitization_run: DigitizationRun,
    table_structure: TableStructure,
    scan_path: Path,
    digitizers: tuple[CellsListDigitizer[Any], ...],
    script_config: AppConfig,
    session: Session,
):

    # cells = [cell.to_ocr_cellspec() for cell in table_structure.cell_structures]

    if not table_structure.cell_structures:
        logging.warning(f"No cells found for table {table_structure.id}")
        return

    # Bad hack
    if cell := next(iter(table_structure.cell_structures), None):
        if (
            any(
                filter(
                    lambda content: cast(CellContent, content).run == digitization_run,
                    cell.cell_contents,
                )
            )
            and not script_config.overwrite
        ):
            return

    # Handle overwrite logic - clear existing contents for this run
    with session.begin_nested():
        for digitizer in digitizers:
            ocr_name = digitizer.ocr.name
            ocr_version = digitizer.ocr.version
            if script_config.overwrite:
                for cell_struct in table_structure.cell_structures:
                    # Delete all contents for this cell and this run using proper ORM
                    stmt = select(CellContent).where(
                        CellContent.cell_id == cell_struct.id,
                        CellContent.run_id == digitization_run.id,
                        CellContent.reader == ocr_name,
                        CellContent.reader_version == ocr_version,
                    )
                    contents_to_delete = session.scalars(stmt).all()
                    for content in contents_to_delete:
                        session.delete(content)
            logging.info(
                f"Running {ocr_name} (v{ocr_version}) on table {table_structure.id}"
            )
            start = datetime.now()

            debug_path = (
                script_config.debug_dir / f"{scan_path.with_suffix('.jpg').name}"
                if script_config.debug_dir
                else None
            )
            if debug_path:
                debug_path.parent.mkdir(parents=True, exist_ok=True)

            scan = read_matlike_image(scan_path)
            results = digitizer(
                scan,
                [cell.to_ocr_cellspec() for cell in table_structure.cell_structures],
                debug_path=debug_path,
            )
            cells_by_position = {
                (cell.row, cell.col): cell for cell in table_structure.cell_structures
            }
            for result in results:
                # Find or create CellContent
                content = CellContent(
                    cell=cells_by_position[
                        (result.cell_spec.row, result.cell_spec.col)
                    ],
                    run=digitization_run,
                    reader=ocr_name,
                    reader_version=ocr_version,
                    content=result.text,
                    confidence=result.confidence,
                    config=None,  # digitizer.ocr.to_dict(),
                )
                session.add(content)

            logging.info(
                f"Ran {ocr_name} (v{ocr_version}) on table {table_structure.id}. It took {(datetime.now() - start).total_seconds()} seconds."
            )
        session.flush()


def main():
    parser = create_argparser()
    args = parser.parse_args()
    script_config = setup_environment(args)
    scan_list: set[str] | None = (
        set(
            pl.read_excel(script_config.scan_list)
            .filter(pl.col("student").is_not_null())["filename"]
            .to_list()
        )
        if script_config.scan_list
        else None
    )

    def scan_in_list(scan_filename: str):
        return scan_list is None or scan_filename in scan_list

    engine = create_engine(script_config.db_uri)
    Base.metadata.create_all(engine)

    # Setup Preprocessing Pipeline as requested
    preprocessor = PreprocessingPipeline(
        "pipeline", [Binarization("binarization"), LineEraser("line_eraser")]
    )

    # Initialize OCRs
    ocr_modules = (
        MultipleOCRModule(
            name="tesserocr",
            version=tesserocr.tesseract_version(),
            ocrs=(
                build_tesserocr_text_ocr("ita"),
                build_tesserocr_number_ocr("ita", extra_whitelist="?»>-_"),
            ),
            filters=[lambda cell: cell.col == 0, lambda cell: cell.col != 0],
        ),
        PaddleOCR(model_name="latin_PP-OCRv5_mobile_rec", kwargs={}),
        MultipleOCRModule(
            name="easyocr",
            version=easyocr.__version__,
            ocrs=(
                EasyOCRParagraph(
                    lang_list=["it"],
                    call_config={"allowlist": f"{ascii_letters}'.-"},
                    name="easyocr",
                    version=easyocr.__version__,
                ),
                EasyOCRWord(
                    lang_list=["it"],
                    call_config={"allowlist": "0123456789-?>»_"},
                    name="easyocr",
                    version=easyocr.__version__,
                ),
            ),
            filters=[lambda cell: cell.col == 0, lambda cell: cell.col != 0],
        ),
    )

    digitizers = tuple(
        [CellsListDigitizer[Any](ocr, preprocessor) for ocr in ocr_modules]
    )

    with Session(engine) as session:  # , session.begin():
        (
            preprocessing_run,
            structure_run,
            digitization_run,
        ) = get_or_init(
            script_config.project_name,
            script_config.preprocessing_run_tag,
            script_config.structure_run_tag,
            script_config.digitization_run_tag,
            session,
        )

        # Get all tables from the given structure run
        tables = sorted(
            structure_run.table_structures,
            key=lambda table: table.preprocessed_scan.filename,
        )
        if args.num:
            tables = tables[: args.num]

        # Map scans to paths
        scan_map = {ps.filename: ps for ps in preprocessing_run.preprocessed_scans}
        scan_paths = find_nested_files(scan_map, script_config.input_path)

        n_processed = 0
        pbar = tqdm(tables, desc="Digitizing tables")
        for table in pbar:
            pscan = table.preprocessed_scan
            if not scan_in_list(pscan.scan.filename):
                pbar.write(f"Scan {pscan.scan.filename} is not in list")
                continue
            pbar.set_description(f"Digitizing {pscan.scan.filename}")

            scan_path = scan_paths.get(pscan)
            if not scan_path:
                pbar.write(
                    f"Scan {pscan.filename} not found in {script_config.input_path}"
                )
                logging.error(f"Scan {pscan.scan.filename} not found")
                continue

            try:
                digitize_and_persist(
                    digitization_run=digitization_run,
                    table_structure=table,
                    scan_path=scan_path,
                    digitizers=digitizers,
                    script_config=script_config,
                    session=session,
                )
                n_processed += 1
            except KeyboardInterrupt:
                pbar.write("Commiting session and exiting gracefully...")
                session.commit()
                sys.exit()
            except Exception as e:
                pbar.write(
                    f"Error digitizing table {table.id} in scan {pscan.scan.filename}: {e}"
                )
                logging.exception(
                    f"Error digitizing table {table.id} in scan {pscan.scan.filename}: {e}"
                )

        pbar.write(f"\nDigitization complete: {n_processed} tables processed.")


def create_argparser() -> argparse.ArgumentParser:
    """Create and configure argument parser for batch processing."""
    parser = argparse.ArgumentParser(
        prog="nivo-reader-digitization",
        description="""Batch digitization of NIVO table images.""",
    )

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
        help="Tag of the structure detection run.",
    )
    _ = parser.add_argument(
        "--digitization-run-tag",
        required=True,
        type=str,
        help="Tag assigned by the user to the current digitization run.",
    )
    _ = parser.add_argument(
        "-p",
        "--project-dir",
        type=Path,
        required=True,
        help="The main directory of the project.",
    )
    _ = parser.add_argument(
        "-d",
        "--debug",
        action="store_true",
        help="Generate debug artifacts.",
    )
    _ = parser.add_argument(
        "-w",
        "--overwrite",
        action="store_true",
        help="Overwrite existing output values in the DB",
    )
    _ = parser.add_argument("--logging-level", type=int, default=logging.INFO)
    _ = parser.add_argument("--num", type=int)
    _ = parser.add_argument("--scan-list", type=Path)
    return parser


if __name__ == "__main__":
    main()
