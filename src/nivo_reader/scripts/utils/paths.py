from pathlib import Path
import logging
from typing import TypeVar
from pydantic import BaseModel, ValidationError
from configurator import Config, default_mergers


def reroute_file(file_path: Path, output_dir: Path, relative_to: Path) -> Path:
    if file_path.is_relative_to(relative_to):
        return output_dir / file_path.relative_to(relative_to)
    raise ValueError(f"File path {file_path} is not relative to {relative_to}")


def discover_files(input_dir: Path, name_pattern: str) -> list[Path]:
    return list(input_dir.rglob(name_pattern))


def _merge_replace(_env, _source, target):
    return target


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


def build_config_stack(
    root,
    model: type[M],
    section: str | None = None,
    config_filename: str = "config.toml",
    base_config: Config | None = None,
) -> dict[Path, M]:
    """
    Traverse the directory tree starting at `root`, collecting the effective
    configuration for every regular file (not the config file itself).

    Returns a mapping: file_path -> merged configuration dict.
    """
    base = base_config or Config()
    file_configs: dict[Path, M] = {}

    def walk(directory: Path, parent_config: Config):
        # 1. Load local TOML config and merge with parent
        config_path = directory / config_filename
        if config_path.is_file():
            local_config = Config.from_path(config_path)  # pyright: ignore[reportUnknownMemberType]
            current_config = merge_configs(parent_config, local_config)
        else:
            current_config = parent_config

        # 2. Assign config to files, recurse into subdirectories
        for entry in directory.iterdir():
            if entry.name == config_filename or entry.suffix == ".toml":
                continue  # skip the config file itself
            if entry.is_file():
                file_config = merge_configs(current_config, load_fconf(entry))
                try:
                    relevant_section = (
                        file_config.data[section]  # pyright: ignore[reportUnknownMemberType]
                        if section is not None
                        else file_config.data  # pyright: ignore[reportUnknownMemberType]
                    )
                    file_configs[entry] = model.model_validate(relevant_section)
                except ValidationError as ve:
                    logging.error(
                        f"validation error for {entry} with {file_config}: {ve}"
                    )
                except KeyError as ke:
                    logging.error(
                        f"key '{section}' not found in config for {entry} with {file_config}: {ke}"
                    )
            elif entry.is_dir():
                walk(entry, current_config)

    walk(Path(root), base)
    return file_configs


def filter_stack(stack, predicate):
    return {k: v for k, v in stack.items() if predicate(k)}  # pyright: ignore[reportUnknownMemberType]
