from collections.abc import Generator
from os import PathLike
from pathlib import Path
import hashlib
import logging
from typing import Any, TypeVar, cast
from pydantic import BaseModel, ValidationError
from configurator import Config, default_mergers


def reroute_file(file_path: Path, output_dir: Path, relative_to: Path) -> Path:
    if file_path.is_relative_to(relative_to):
        return output_dir / file_path.relative_to(relative_to)
    raise ValueError(f"File path {file_path} is not relative to {relative_to}")


def discover_files(input_dir: Path, name_pattern: str) -> list[Path]:
    return list(input_dir.rglob(name_pattern))


def _merge_replace(_env: Any, _source: Any, _target: Any):
    raise Exception("This is bugged. Don't use")


def merge_configs(parent: Config, local: Config) -> Config:
    p = parent.clone()
    p.merge(  # pyright: ignore[reportUnknownMemberType]
        local, mergers=default_mergers + {list: _merge_replace}
    )
    return p


def load_fconf(path: Path):
    fconf = path.with_suffix(".toml")
    return Config.from_path(fconf) if fconf.is_file() else Config()  # pyright: ignore[reportUnknownMemberType]


M = TypeVar("M", bound=BaseModel)
P = TypeVar("P")


def build_config_dict(
    root: PathLike[str] | Path,
    model: type[M],
    start: PathLike[str] | Path | None = None,
    section: str | None = None,
    config_filename: str = "config.toml",
    base_config: Config | None = None,
):
    return dict(
        build_config_stack(root, model, start, section, config_filename, base_config)
    )


def build_config_stack(
    root: PathLike[str] | Path,
    model: type[M],
    start: PathLike[str] | Path | None = None,
    section: str | None = None,
    config_filename: str = "config.toml",
    base_config: Config | None = None,
) -> Generator[tuple[Path, M | ValidationError | KeyError], None, None]:
    """
    Traverse the directory tree starting at `root`, collecting the effective
    configuration for every regular file by merging nested dictionaries.

    Returns a mapping: file_path -> merged configuration dict.
    """
    base = base_config or Config()
    # file_configs: dict[Path, M] = {}
    root = Path(root)
    if start is None:
        start = root

    def walk(
        path: Path, parent_config: Config
    ) -> Generator[tuple[Path, M | ValidationError | KeyError], None, None]:
        if not path.name.startswith("."):
            # Nodes
            if path.is_dir():
                # Update the config
                config_path = path / config_filename
                if config_path.is_file():
                    local_overrides = Config.from_path(config_path)  # pyright: ignore[reportUnknownMemberType]
                    dir_config = merge_configs(parent_config, local_overrides)
                else:
                    dir_config = parent_config

                for entry in path.iterdir():
                    yield from walk(entry, dir_config)
            # Leaves
            elif (
                path.is_file()
                and path.is_relative_to(start)
                and not path.suffix == ".toml"
            ):
                file_config = merge_configs(parent_config, load_fconf(path))
                try:
                    file_config = cast(dict[str, Any], file_config.data)
                    relevant_section = (
                        file_config[section] if section is not None else file_config
                    )
                    yield path, model.model_validate(relevant_section)
                except ValidationError as ve:
                    logging.error(
                        f"validation error for {path} with {file_config}: {ve}"
                    )
                    yield path, ve
                except KeyError as ke:
                    logging.exception(
                        f"key '{section}' not found in config for {path} with {file_config}: {ke}"
                    )
                    yield path, ke

    return walk(root, base)


def mkopath(
    input_path: Path,
    output_dir: Path,
    new_suffix: str | None = None,
    mkdir: bool = False,
) -> Path:
    """Utility.

    Args:
        output_dir: Output directory for Excel files
        image_path: Path to the input image

    Returns:
        Path to the output Excel file
    """
    newpath = output_dir / input_path.name
    if new_suffix:
        newpath = newpath.with_suffix(new_suffix)
    if mkdir:
        newpath.mkdir(parents=True, exist_ok=True)
    return newpath


def get_sha256(filepath: Path) -> str:
    sha256_hash = hashlib.sha256(usedforsecurity=False)
    with open(filepath, "rb") as f:
        for byte_block in iter(lambda: f.read(8192), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def find_nested_files[T](mapping: dict[str, T], root: Path) -> dict[T, Path]:
    out: dict[T, Path] = {}
    for path, _, files in root.walk():
        for file in files:
            if file in mapping:
                out[mapping[file]] = path / file
    return out
