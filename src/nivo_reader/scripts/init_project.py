import argparse
import logging
from pathlib import Path

from pydantic import BaseModel, DirectoryPath
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from tqdm import tqdm

from nivo_reader.models.db import Base, Project, Scan
from nivo_reader.scripts.utils.paths import get_sha256


class AppConfig(BaseModel):
    project_dir: DirectoryPath
    project_name: str
    project_description: str
    db_uri: str
    image_formats: set[str] = {"png", "jpg", "jpeg", "gif"}
    logging_level: int = logging.INFO


def setup_environment(args: argparse.Namespace) -> AppConfig:
    cli_params = {
        k: v
        for k, v in vars(args).items()
        if v is not None and k in AppConfig.model_fields
    }

    project_dir = Path(cli_params.get("project_dir", args.project_dir))

    cli_params["db_uri"] = f"sqlite:///{project_dir}/db.sqlite"

    script_config = AppConfig(**cli_params)

    logging.basicConfig(
        level=script_config.logging_level,
        format="[%(asctime)s][%(levelname)s]%(name)s - %(message)s",
    )

    return script_config


def create_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="nivo-reader-init",
        description="Initialize a NIVO reader project and index input scan files.",
    )

    _ = parser.add_argument(
        "-p",
        "--project-dir",
        type=Path,
        required=True,
        help="The main directory of the project.",
    )
    _ = parser.add_argument(
        "-n",
        "--project-name",
        type=str,
        required=True,
        help="The name of the project.",
    )
    _ = parser.add_argument(
        "--project-description",
        type=str,
        default="",
        help="Description of the project.",
    )
    _ = parser.add_argument(
        "--image-formats",
        type=lambda s: set(s.split(",")),
        default="png,jpg,jpeg,gif",
        help="Comma-separated list of image file formats to process (default: png,jpg,jpeg,gif).",
    )
    _ = parser.add_argument(
        "--logging-level",
        type=int,
        default=logging.INFO,
        help="Logging level (default: 20 for INFO).",
    )

    return parser


def is_valid_file(path: Path, formats: set[str]) -> bool:
    return path.is_file() and (path.suffix.strip(".").lower() in formats)


def main():
    parser = create_argparser()
    args = parser.parse_args()
    script_config = setup_environment(args)

    engine = create_engine(script_config.db_uri)
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)

    input_dir = script_config.project_dir / "00_input"
    image_paths = [
        p for p in input_dir.rglob("*") if is_valid_file(p, script_config.image_formats)
    ]

    with Session() as session:
        # Get or create project
        project = (
            session.query(Project)
            .filter_by(name=script_config.project_name)
            .one_or_none()
        )
        if not project:
            logging.info(f"Creating new project '{script_config.project_name}'")
            project = Project(
                name=script_config.project_name,
                description=script_config.project_description,
            )
            existing_scans: set[str] = set()
        else:
            logging.info(f"Found existing project '{script_config.project_name}'")
            existing_scans = set(map(lambda s: s.filename, project.scans))

        added_count = 0
        pbar = tqdm(image_paths, desc="Scanning input files")
        for image_path in pbar:
            # Check if scan with this filename already exists in this project
            if image_path.name not in existing_scans:
                scan_hash = get_sha256(image_path)
                scan = Scan(
                    sha256Hash=scan_hash,
                    filename=image_path.name,
                )
                project.scans.append(scan)
                added_count += 1

        session.add(project)
        session.commit()
        logging.info(f"Project initialization complete. Added {added_count} new scans.")


if __name__ == "__main__":
    main()
