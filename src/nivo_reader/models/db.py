from datetime import datetime
from pathlib import Path
from typing import Any, final

from sqlalchemy import JSON, Float, ForeignKey, String, UniqueConstraint, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

from nivo_reader.lib.common import BoundingBox
from nivo_reader.modules.table_digitization.base import OCRCellSpec


class Base(DeclarativeBase):
    pass


@final
class Project(Base):
    __tablename__ = "project"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String, unique=True, nullable=False)
    description: Mapped[str | None] = mapped_column(String, nullable=True)

    scans: Mapped[list["Scan"]] = relationship(back_populates="project")
    preprocessing_runs: Mapped[list["PreprocessingRun"]] = relationship(
        back_populates="project"
    )
    structure_runs: Mapped[list["StructureRun"]] = relationship(
        back_populates="project"
    )
    digitization_runs: Mapped[list["DigitizationRun"]] = relationship(
        back_populates="project"
    )


@final
class Scan(Base):
    __tablename__ = "scan"
    __table_args__ = (UniqueConstraint("project_id", "filename"),)

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    sha256Hash: Mapped[str] = mapped_column(
        String, nullable=False, doc="SHA256 hash of the image file"
    )
    filename: Mapped[str] = mapped_column(
        String,
        nullable=False,
        doc="Name portion of the file path. Must be unique within the project.",
    )
    project_id: Mapped[int] = mapped_column(ForeignKey("project.id"), nullable=False)

    project: Mapped["Project"] = relationship(back_populates="scans")
    preprocessed_scans: Mapped[list["PreprocessedScan"]] = relationship(
        back_populates="scan"
    )

    def find_path(self, output_dir: Path):
        return next(
            (path for path in output_dir.rglob(f"**/{self.filename}")),
            None,
        )


@final
class PreprocessingRun(Base):
    __tablename__ = "preprocessingRun"
    __table_args__ = (UniqueConstraint("project_id", "tag"),)

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    project_id: Mapped[int] = mapped_column(ForeignKey("project.id"), nullable=False)
    tag: Mapped[str] = mapped_column(
        String,
        nullable=False,
        doc="Unique tag for this preprocessing run within the project.",
    )
    createdAt: Mapped[datetime] = mapped_column(
        server_default=func.now(), nullable=False
    )
    updatedAt: Mapped[datetime] = mapped_column(
        server_default=func.now(), onupdate=func.now(), nullable=False
    )

    project: Mapped["Project"] = relationship(back_populates="preprocessing_runs")
    preprocessed_scans: Mapped[list["PreprocessedScan"]] = relationship(
        back_populates="run", cascade="all, delete, delete-orphan"
    )


@final
class PreprocessedScan(Base):
    __tablename__ = "preprocessedScan"
    __table_args__ = (UniqueConstraint("scan_id", "run_id"),)

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    scan_id: Mapped[int] = mapped_column(
        ForeignKey("scan.id"), nullable=False, doc="Scan that was preprocessed"
    )
    run_id: Mapped[int] = mapped_column(
        ForeignKey("preprocessingRun.id"),
        nullable=False,
    )
    sha256Hash: Mapped[str] = mapped_column(
        String, nullable=False, doc="SHA256 hash of the preprocessed scan"
    )
    config: Mapped[dict[str, Any]] = mapped_column(
        JSON, nullable=False, doc="Configuration used by the preprocessing pipeline"
    )

    scan: Mapped["Scan"] = relationship(back_populates="preprocessed_scans")
    run: Mapped["PreprocessingRun"] = relationship(back_populates="preprocessed_scans")
    table_structures: Mapped[list["TableStructure"]] = relationship(
        back_populates="preprocessed_scan", cascade="all, delete, delete-orphan"
    )

    @property
    def filename(self):
        return self.filename_from(self.scan)

    @classmethod
    def filename_from(cls, scan: Scan):
        return Path(scan.filename).with_suffix(".png").name

    @classmethod
    def find_path(cls, scan: Scan, output_dir: Path):
        return next(
            (path for path in output_dir.rglob(f"{cls.filename_from(scan)}")),
            None,
        )


@final
class StructureRun(Base):
    __tablename__ = "structureRun"
    __table_args__ = (UniqueConstraint("project_id", "tag"),)

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    project_id: Mapped[int] = mapped_column(ForeignKey("project.id"), nullable=False)

    tag: Mapped[str] = mapped_column(String, nullable=False)
    createdAt: Mapped[datetime] = mapped_column(
        server_default=func.now(), nullable=False
    )
    updatedAt: Mapped[datetime] = mapped_column(
        server_default=func.now(), onupdate=func.now(), nullable=False
    )

    project: Mapped["Project"] = relationship(back_populates="structure_runs")
    table_structures: Mapped[list["TableStructure"]] = relationship(
        back_populates="run", cascade="all, delete, delete-orphan"
    )


@final
class TableStructure(Base):
    __tablename__ = "tableStructure"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    preprocessedScan_id: Mapped[int] = mapped_column(
        ForeignKey("preprocessedScan.id"), nullable=False
    )
    config: Mapped[dict[str, Any]] = mapped_column(
        JSON,
        nullable=False,
        doc="Configuration used by the structure detection pipeline",
    )

    bbox: Mapped[dict[str, Any]] = mapped_column(
        JSON,
        nullable=False,
        doc="Bounding box of the table within the preprocessed scan",
    )
    header: Mapped[dict[str, Any] | None] = mapped_column(
        JSON, nullable=True, doc="Header bounding box and details"
    )
    index: Mapped[dict[str, Any] | None] = mapped_column(
        JSON, nullable=True, doc="Index bounding box and details"
    )
    content: Mapped[dict[str, Any]] = mapped_column(
        JSON, nullable=False, doc="Content bounding box and details"
    )
    nrows: Mapped[int] = mapped_column(
        nullable=False, doc="Number of rows in the table"
    )
    ncols: Mapped[int] = mapped_column(
        nullable=False, doc="Number of columns in the table"
    )
    run_id: Mapped[int] = mapped_column(ForeignKey("structureRun.id"), nullable=False)

    preprocessed_scan: Mapped["PreprocessedScan"] = relationship(
        back_populates="table_structures"
    )
    run: Mapped["StructureRun"] = relationship(back_populates="table_structures")
    cell_structures: Mapped[list["CellStructure"]] = relationship(
        back_populates="table", cascade="all, delete, delete-orphan"
    )


@final
class CellStructure(Base):
    __tablename__ = "cellStructure"
    __table_args__ = (UniqueConstraint("table_id", "row", "col"),)

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    table_id: Mapped[int] = mapped_column(
        ForeignKey("tableStructure.id"), nullable=False
    )
    config: Mapped[dict[str, Any]] = mapped_column(
        JSON,
        nullable=False,
        doc="Configuration used by the cell structure detection pipeline",
    )
    bbox: Mapped[dict[str, Any]] = mapped_column(
        JSON,
        nullable=False,
        doc="Bounding box of the cell within the preprocessed scan",
    )
    row: Mapped[int] = mapped_column(nullable=False, doc="Row index of the cell")
    col: Mapped[int] = mapped_column(nullable=False, doc="Column index of the cell")

    table: Mapped["TableStructure"] = relationship(back_populates="cell_structures")
    cell_contents: Mapped[list["CellContent"]] = relationship(
        back_populates="cell", cascade="all, delete, delete-orphan"
    )

    def to_ocr_cellspec(self) -> OCRCellSpec[BoundingBox]:
        return OCRCellSpec(
            bbox=BoundingBox(**self.bbox),
            col=self.col,
            row=self.row,
        )


@final
class DigitizationRun(Base):
    __tablename__ = "digitizationRun"
    __table_args__ = (UniqueConstraint("project_id", "tag"),)

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    project_id: Mapped[int] = mapped_column(ForeignKey("project.id"), nullable=False)
    tag: Mapped[str] = mapped_column(String, nullable=False)
    createdAt: Mapped[datetime] = mapped_column(
        server_default=func.now(), nullable=False
    )
    updatedAt: Mapped[datetime] = mapped_column(
        server_default=func.now(), onupdate=func.now(), nullable=False
    )

    project: Mapped["Project"] = relationship(back_populates="digitization_runs")
    cell_contents: Mapped[list["CellContent"]] = relationship(
        back_populates="run", cascade="all, delete, delete-orphan"
    )


# @final
# class Reader(Base):
#     __tablename__ = "reader"
#     id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
#     name: Mapped[str] = mapped_column(String, nullable=False, unique=True)
#     version: Mapped[str] = mapped_column(String, nullable=False)
#     description: Mapped[str] = mapped_column(String, nullable=True)
#     infos: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=True)
#     cell_contents: Mapped[list["CellContent"]] = relationship(back_populates="reader")


@final
class CellContent(Base):
    __tablename__ = "cellContent"
    __table_args__ = (UniqueConstraint("cell_id", "run_id", "reader"),)

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    cell_id: Mapped[int] = mapped_column(ForeignKey("cellStructure.id"), nullable=False)
    run_id: Mapped[int] = mapped_column(
        ForeignKey("digitizationRun.id"), nullable=False
    )
    reader: Mapped[str] = mapped_column(String, nullable=False)
    reader_version: Mapped[str] = mapped_column(String, nullable=False)
    config: Mapped[dict[str, Any]] = mapped_column(
        JSON,
        nullable=False,
        doc="Configuration used by the digitization pipeline",
    )
    content: Mapped[str | None] = mapped_column(String, nullable=True)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)

    cell: Mapped["CellStructure"] = relationship(back_populates="cell_contents")
    run: Mapped["DigitizationRun"] = relationship(back_populates="cell_contents")
