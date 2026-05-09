from datetime import datetime
from typing import Any, final

from sqlalchemy import JSON, ForeignKey, String, UniqueConstraint, func
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
    __table_args__ = (
        UniqueConstraint("project_id", "filename", name="uq_scan_project_filename"),
    )

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    sha256Hash: Mapped[str] = mapped_column(String, nullable=False)
    filename: Mapped[str] = mapped_column(String, nullable=False)
    project_id: Mapped[int] = mapped_column(ForeignKey("project.id"), nullable=False)

    project: Mapped["Project"] = relationship(back_populates="scans")
    preprocessed_scans: Mapped[list["PreprocessedScan"]] = relationship(
        back_populates="scan"
    )


@final
class PreprocessingRun(Base):
    __tablename__ = "preprocessingRun"
    __table_args__ = (UniqueConstraint("project_id", "tag"),)

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    project_id: Mapped[int] = mapped_column(ForeignKey("project.id"), nullable=False)
    # The config json field refers to the PipelineConfig model in the respective script
    config: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    tag: Mapped[str] = mapped_column(String, nullable=False)
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
    scan_id: Mapped[int] = mapped_column(ForeignKey("scan.id"), nullable=False)
    run_id: Mapped[int] = mapped_column(
        ForeignKey("preprocessingRun.id"), nullable=False
    )
    sha256Hash: Mapped[str] = mapped_column(String, nullable=False)

    scan: Mapped["Scan"] = relationship(back_populates="preprocessed_scans")
    run: Mapped["PreprocessingRun"] = relationship(back_populates="preprocessed_scans")
    table_structures: Mapped[list["TableStructure"]] = relationship(
        back_populates="preprocessed_scan"
    )


@final
class StructureRun(Base):
    __tablename__ = "structureRun"
    __table_args__ = (UniqueConstraint("project_id", "tag"),)

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    project_id: Mapped[int] = mapped_column(ForeignKey("project.id"), nullable=False)
    # The config json field refers to the PipelineConfig model in the respective script
    config: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
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
    bbox: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    header: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    index: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    content: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    nRows: Mapped[int] = mapped_column(nullable=False)
    nCols: Mapped[int] = mapped_column(nullable=False)
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
    row: Mapped[int] = mapped_column(nullable=False)
    col: Mapped[int] = mapped_column(nullable=False)

    table: Mapped["TableStructure"] = relationship(back_populates="cell_structures")
    cell_contents: Mapped[list["CellContent"]] = relationship(back_populates="cell")


@final
class DigitizationRun(Base):
    __tablename__ = "digitizationRun"
    __table_args__ = (UniqueConstraint("project_id", "tag"),)

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    project_id: Mapped[int] = mapped_column(ForeignKey("project.id"), nullable=False)
    # The config json field refers to the PipelineConfig model in the respective script
    config: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
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
    cellType: Mapped[str] = mapped_column(String, nullable=False)
    content: Mapped[str | None] = mapped_column(String, nullable=True)
    run_id: Mapped[int] = mapped_column(
        ForeignKey("digitizationRun.id"), nullable=False
    )

    cell: Mapped["CellStructure"] = relationship(back_populates="cell_contents")
    run: Mapped["DigitizationRun"] = relationship(back_populates="cell_contents")
