from datetime import datetime
from pathlib import Path
from typing import Any, final

from sqlalchemy import JSON, Enum, ForeignKey, String, UniqueConstraint, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


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
        back_populates="run"
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
        back_populates="preprocessed_scan"
    )

    @classmethod
    def filename(cls, scan: Scan):
        return Path(scan.filename).with_suffix(".png").name

    @classmethod
    def find_path(cls, scan: Scan, output_dir: Path):
        return next(
            (path for path in output_dir.rglob(f"**/{cls.filename(scan)}")),
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
        back_populates="run"
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
        back_populates="table"
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
    row: Mapped[int] = mapped_column(nullable=False, doc="Row index of the cell")
    col: Mapped[int] = mapped_column(nullable=False, doc="Column index of the cell")

    table: Mapped["TableStructure"] = relationship(back_populates="cell_structures")
    cell_contents: Mapped[list["CellContent"]] = relationship(back_populates="cell")


@final
class DigitizationRun(Base):
    __tablename__ = "digitizationRun"
    __table_args__ = (UniqueConstraint("project_id", "tag"),)

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    project_id: Mapped[int] = mapped_column(ForeignKey("project.id"), nullable=False)
    tag: Mapped[str] = mapped_column(String, nullable=False)
    ocr: Mapped[str | None] = mapped_column(String, nullable=True)
    createdAt: Mapped[datetime] = mapped_column(
        server_default=func.now(), nullable=False
    )
    updatedAt: Mapped[datetime] = mapped_column(
        server_default=func.now(), onupdate=func.now(), nullable=False
    )

    project: Mapped["Project"] = relationship(back_populates="digitization_runs")
    cell_contents: Mapped[list["CellContent"]] = relationship(back_populates="run")


@final
class CellContent(Base):
    __tablename__ = "cellContent"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    cell_id: Mapped[int] = mapped_column(ForeignKey("cellStructure.id"), nullable=False)
    config: Mapped[dict[str, Any]] = mapped_column(
        JSON,
        nullable=False,
        doc="Configuration used by the digitization pipeline",
    )
    cellType: Mapped[str] = mapped_column(Enum("numeric", "text"), nullable=False)
    content: Mapped[str | None] = mapped_column(String, nullable=True)
    run_id: Mapped[int] = mapped_column(
        ForeignKey("digitizationRun.id"), nullable=False
    )

    cell: Mapped["CellStructure"] = relationship(back_populates="cell_contents")
    run: Mapped["DigitizationRun"] = relationship(back_populates="cell_contents")
