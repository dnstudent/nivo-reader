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
import logging
import hashlib
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
from sqlalchemy import String, create_engine, DateTime, func, Integer, JSON, ForeignKey
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker
from tqdm import tqdm

from nivo_reader.lib.images import read_matlike_image
from nivo_reader.modules.structure_detection.base import (
    StructureDetector,
    StructureResult,
)
from nivo_reader.modules.structure_detection.nivo_structure_detection import (
    NivoStructureDetection,
    NivoStructureDetectionConfig as PipelineConfig,
)
from nivo_reader.scripts.utils.paths import build_config_stack


STEP_NUMBER = "02"
STEP_NAME = "table_structure"


class Base(DeclarativeBase):
    pass


def get_structure_models(prefix: str = ""):
    class StructureTable(Base):
        __tablename__: str = f"{prefix}table" if prefix else "table"
        __table_args__: dict[str, Any] = {"extend_existing": True}
        id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
        preprocessingRunTag: Mapped[str] = mapped_column(String, nullable=False)
        structureRunTag: Mapped[str] = mapped_column(String, nullable=False)
        scanName: Mapped[str] = mapped_column(String, nullable=False)
        scanChecksum: Mapped[str] = mapped_column(String, nullable=False)
        bbox: Mapped[dict[str, int]] = mapped_column(JSON, nullable=False)
        header: Mapped[dict[str, int] | None] = mapped_column(JSON, nullable=True)
        index: Mapped[dict[str, int] | None] = mapped_column(JSON, nullable=True)
        content: Mapped[dict[str, int]] = mapped_column(JSON, nullable=False)
        num_rows: Mapped[int] = mapped_column(Integer, nullable=False)
        num_cols: Mapped[int] = mapped_column(Integer, nullable=False)
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

    class StructureCell(Base):
        __tablename__: str = f"{prefix}cell" if prefix else "cell"
        __table_args__: dict[str, Any] = {"extend_existing": True}
        table_id: Mapped[int] = mapped_column(
            Integer, ForeignKey(f"{prefix}table.id"), primary_key=True
        )
        row: Mapped[int] = mapped_column(Integer, primary_key=True)
        column: Mapped[int] = mapped_column(Integer, primary_key=True)
        cell_region: Mapped[dict[str, int]] = mapped_column(JSON, nullable=False)
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

    return StructureTable, StructureCell


class AppConfig(BaseModel):
    db_uri: str
    table_prefix: str = "structure_"
    preprocessing_run_tag: str
    structure_run_tag: str
    project_dir: DirectoryPath
    debug_dir: Path | None = None
    image_formats: set[str] = {"png"}
    overwrite: bool = False
    pipeline_config_fname: str = "config.toml"
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


def get_file_checksum(path: Path) -> str:
    sha256_hash = hashlib.sha256()
    with open(path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def drop_existing_structure(
    session: Any,
    StructureTable: Any,
    StructureCell: Any,
    structure_run_tag: str,
    scan_name: str,
) -> None:
    """Drop existing tables and their cells for a specific scan and run tag."""
    tables_to_del = (
        session.query(StructureTable.id)
        .filter_by(
            structureRunTag=structure_run_tag,
            scanName=scan_name,
        )
        .all()
    )

    if not tables_to_del:
        return

    table_ids = [t.id for t in tables_to_del]

    _ = (
        session.query(StructureCell)
        .filter(StructureCell.table_id.in_(table_ids))
        .delete(synchronize_session=False)
    )
    _ = (
        session.query(StructureTable)
        .filter(StructureTable.id.in_(table_ids))
        .delete(synchronize_session=False)
    )


def main():
    parser = create_argparser()
    args = parser.parse_args()
    script_config = setup_environment(args)

    engine = create_engine(script_config.db_uri)
    StructureTable, StructureCell = get_structure_models(script_config.table_prefix)
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)

    def is_valid_file(path: Path) -> bool:
        return path.is_file() and (
            path.suffix.strip(".") in script_config.image_formats
        )

    iostack = filter(
        lambda entry: is_valid_file(entry[0]),
        build_config_stack(
            root=script_config.project_dir,
            start=script_config.input_path,
            model=PipelineConfig,
            config_filename=script_config.pipeline_config_fname,
        ),
    )

    scan_config_stack = sorted(
        iostack,
        key=lambda x: x[0],
    )

    detector = NivoStructureDetection()

    pbar = tqdm(scan_config_stack, desc="Processing scans")
    with Session() as session:
        for scan_path, scan_config in pbar:
            logging.debug(f"Reading from {scan_path} with conf {scan_config}")
            if scan_config is None:
                pbar.write(
                    f"✗ Error processing {scan_path.relative_to(script_config.project_dir)}: the configuration is invalid. Check the log."
                )
                continue

            pbar.set_description(f"Processing {scan_path.name}")

            try:
                # Check for existing entries
                exists = (
                    session.query(StructureTable)
                    .filter_by(
                        structureRunTag=script_config.structure_run_tag,
                        scanName=scan_path.name,
                    )
                    .first()
                    is not None
                )

                if exists and not script_config.overwrite:
                    pbar.write(
                        f"⏭ Skipped: {scan_path.relative_to(script_config.project_dir)} (already exists)"
                    )
                    continue

                if exists and script_config.overwrite:
                    # Delete existing entries to avoid primary key conflicts and stale data
                    drop_existing_structure(
                        session,
                        StructureTable,
                        StructureCell,
                        script_config.structure_run_tag,
                        scan_path.name,
                    )

                scan = read_matlike_image(scan_path, grayscale=False)
                structure, _ = detect_structure(
                    scan,
                    scan_config,
                    detector,
                    script_config.debug_dir / scan_path.with_suffix(".jpg").name
                    if script_config.debug_dir
                    else None,
                )

                if len(structure.tables) == 0:
                    raise ValueError("No tables detected in the scan.")

                # Calculate scan checksum
                scan_checksum = get_file_checksum(scan_path)

                for table in structure.tables:
                    header_spec = table.header_spec
                    content_spec = table.content_spec
                    index_spec = table.index_spec

                    # Insert/Update Table
                    table_result = StructureTable(
                        preprocessingRunTag=script_config.preprocessing_run_tag,
                        structureRunTag=script_config.structure_run_tag,
                        scanName=scan_path.name,
                        scanChecksum=scan_checksum,
                        bbox=table.full_region.to_dict(),
                        header=header_spec.to_dict() if header_spec else None,
                        index=index_spec.to_dict() if index_spec else None,
                        content=content_spec.to_dict(),
                        num_rows=content_spec.nrows,
                        num_cols=content_spec.ncols,
                    )
                    session.add(table_result)
                    session.flush()  # To get the table_result.id

                    # Insert/Update Cells
                    for cell in content_spec.cells:
                        if cell.cell_region is None:
                            continue
                        cell_result = StructureCell(
                            table_id=table_result.id,
                            row=cell.row,
                            column=cell.column,
                            cell_region=cell.cell_region.to_dict(),
                        )
                        session.add(cell_result)

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
        prog="nivo-reader-structure-detection",
        description="""Batch structure detection of NIVO table images.""",
    )

    # Input/Output arguments
    _ = parser.add_argument(
        "-i",
        "--input-path",
        required=False,
        type=Path,
        help="Subset of the images to process. Could be a directory or a single image. Default is the project root's 01_preprocess/preprocessing_run_tag.",
    )
    _ = parser.add_argument(
        "--table-prefix",
        type=str,
        default="structure_",
        help="Prefix for the table and cell tables to store the results.",
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
