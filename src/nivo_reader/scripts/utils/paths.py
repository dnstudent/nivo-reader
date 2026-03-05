from pathlib import Path


def reroute_file(file_path: Path, output_dir: Path, relative_to: Path) -> Path:
    if file_path.is_relative_to(relative_to):
        return output_dir / file_path.relative_to(relative_to)
    raise ValueError(f"File path {file_path} is not relative to {relative_to}")


def discover_files(input_dir: Path, name_pattern: str) -> list[Path]:
    return list(input_dir.rglob(name_pattern))
