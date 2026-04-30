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
import warnings
from itertools import chain
from pathlib import Path
import tomllib
import json
from typing import Any

import numpy as np
import polars as pl
from PIL import Image
from tqdm import tqdm

from nivo_reader.nivo_reader import read_nivo_table
from nivo_reader.scripts.utils.paths import discover_files


# Suppress pin_memory warnings from PyTorch/EasyOCR
warnings.filterwarnings("ignore", message=".*pin_memory.*")
logging.getLogger("paddlex").setLevel(logging.ERROR)
logging.getLogger("paddle").setLevel(logging.ERROR)

DEFAULT_CONFIG = {
    "station_char_shape": [12, 10],
    "number_char_shape": [12, 20],
    "roi_padding": 3,
    "nchars_threshold": 20,
    "extra_width": 6,
    "ocr_engines": ["tesserocr", "easyocr", "paddleocr"],
    "multi_row_station_names": False,
}

_OCR_CACHE = {}


def get_ocrs(ocr_engines: list[str]) -> dict[str, Any]:
    ocrs = {}
    for engine in ocr_engines:
        if engine in _OCR_CACHE:
            ocrs[engine] = _OCR_CACHE[engine]
        else:
            if engine == "easyocr":
                import easyocr

                _OCR_CACHE["easyocr"] = easyocr.Reader(lang_list=["it"])
                ocrs["easyocr"] = _OCR_CACHE["easyocr"]
            elif engine == "paddleocr":
                import paddleocr

                _OCR_CACHE["paddleocr"] = paddleocr.TextRecognition(
                    model_name="latin_PP-OCRv5_mobile_rec"
                )
                ocrs["paddleocr"] = _OCR_CACHE["paddleocr"]
            elif engine == "tesseract":
                _OCR_CACHE["tesseract"] = None
                ocrs["tesseract"] = None
            elif engine == "tesserocr":
                import tesserocr as to

                _OCR_CACHE["tesserocr"] = to.PyTessBaseAPI(
                    path="/opt/homebrew/opt/tesseract/share/tessdata/",
                    lang="ita",
                    psm=to.PSM.AUTO,
                )
                ocrs["tesserocr"] = _OCR_CACHE["tesserocr"]
    return ocrs


def discover_configurations(images_dir: Path) -> dict[Path, dict[str, Any]]:
    """Find all *.toml configurations in the subtrees."""
    configs: dict[Path, dict[str, Any]] = {}
    for toml_path in sorted(images_dir.rglob("*.toml")):
        try:
            with open(toml_path, "rb") as f:
                toml_config = tomllib.load(f)
                # Normalize keys (replace '-' with '_')
                normalized_config = {
                    k.replace("-", "_"): v for k, v in toml_config.items()
                }
                configs[toml_path.parent] = normalized_config
        except Exception as e:
            logging.warning(f"Failed to read config %s: %s", toml_path, e)
    return configs


def load_image(image_path: Path) -> np.ndarray:
    """Load image from file and convert to BGR.

    Args:
        image_path: Path to image file

    Returns:
        Image in BGR format

    Raises:
        ValueError: If image cannot be loaded
    """
    try:
        # Reverse the RGB channels to get BGR for OpenCV
        return np.array(Image.open(image_path).convert("RGB"))[:, :, ::-1].copy()
    except Exception as e:
        raise ValueError(f"Could not load image {image_path}: {e}")


def scan_output_dir(
    root_output_dir: Path, scan_input_path: Path, scans_dir: Path
) -> Path:
    """Generate output Excel file path for an image.

    Args:
        output_dir: Output directory for Excel files
        image_path: Path to the input image
        images_dir: Base directory containing images

    Returns:
        Path to the output Excel file
    """
    return root_output_dir / scan_input_path.relative_to(scans_dir)


def compose_debug_dir(debug_dir: Path, image_path: Path, images_dir: Path) -> Path:
    """Generate debug directory path for an image.

    Args:
        debug_dir: Base directory for debug artifacts
        image_path: Path to the input image
        images_dir: Base directory containing images

    Returns:
        Path to the debug directory for this image
    """
    return debug_dir / image_path.relative_to(images_dir) / ""


def extract_digitizations(
    image_paths: list[Path],
    clips: tuple[int, int, int, int],
    table_shape: tuple[int, int],
    ocr_engines: list[str],
    multi_row_station_names: bool,
    from_extracted_structure: bool,
    station_char_shape: tuple[int, int] = (12, 10),
    number_char_shape: tuple[int, int] = (12, 20),
    roi_padding: int = 3,
    nchars_threshold: int = 20,
    extra_width: int = 6,
    debug_base_dir: Path | None = None,
    base_images_dir: Path | None = None,
) -> dict[Path, pl.DataFrame]:
    """Process multiple NIVO images and return their digitizations."""
    ocrs = get_ocrs(ocr_engines)

    results = {}
    for img_path in image_paths:
        try:
            # Generate debug directory if requested
            img_debug_dir = None
            if debug_base_dir and base_images_dir:
                img_debug_dir = compose_debug_dir(
                    debug_base_dir, img_path, base_images_dir
                )

            # Process image
            raw_digitization = read_nivo_table(
                load_image(img_path),
                clips,
                table_shape,
                ocrs,
                multi_row_station_names,
                from_extracted_structure,
                station_char_shape,
                number_char_shape,
                roi_padding,
                nchars_threshold,
                extra_width,
                img_debug_dir,
            )
            results[img_path] = raw_digitization

        except Exception as e:
            logging.exception(f"Error processing {img_path}: {e}")
            # You might want to skip or raise. For now we skip just like before.

    return results


def digitize_scans_batch(
    images_dir: Path,
    image_paths: list[Path],
    output_dir: Path,
    debug_dir: Path | None,
    overwrite: bool,
    config: dict[str, Any],
) -> None:
    """Batch process multiple NIVO images and write to output files."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if debug_dir:
        debug_dir = Path(debug_dir)
        debug_dir.mkdir(parents=True, exist_ok=True)

    # Filter images to process
    if not overwrite:
        images_to_process = [
            image_path
            for image_path in image_paths
            if not scan_output_dir(output_dir, image_path, images_dir).exists()
        ]
    else:
        images_to_process = image_paths

    if not images_to_process:
        logging.warning("No images to process")
        return

    logging.info(f"Processing {len(images_to_process)} images...")

    if "clips" not in config or "table_shape" not in config:
        logging.error(
            "✗ Error: 'clips' and 'table_shape' must be provided via CLI or TOML config."
        )
        return

    for img_path in tqdm(
        images_to_process, desc="Processing images"
    ):  # pyrefly: ignore
        results = extract_digitizations(
            image_paths=[img_path],
            clips=tuple(config["clips"]),
            table_shape=tuple(config["table_shape"]),
            ocr_engines=config["ocr_engines"],
            multi_row_station_names=config["multi_row_station_names"],
            from_extracted_structure=config["from_extracted_structure"],
            station_char_shape=tuple(config["station_char_shape"]),
            number_char_shape=tuple(config["number_char_shape"]),
            roi_padding=config["roi_padding"],
            nchars_threshold=config["nchars_threshold"],
            extra_width=config["extra_width"],
            debug_base_dir=debug_dir,
            base_images_dir=images_dir,
        )

        if img_path in results:
            sod = scan_output_dir(output_dir, img_path, images_dir)
            sod.mkdir(parents=True, exist_ok=True)
            results[img_path].write_json(sod / "raw_digitization.json")
            tqdm.write(f"✓ Processed: {img_path.relative_to(images_dir)}")
        else:
            tqdm.write(f"✗ Error processing {img_path.relative_to(images_dir)}")


def main():
    parser = create_argparser()
    args = parser.parse_args()

    debug_dir: str | None = args.debug_dir
    images_dir: Path = args.images_dir
    output_dir: str = args.output_dir
    overwrite: bool = args.overwrite

    cli_config = {}
    if args.clips is not None:
        cli_config["clips"] = args.clips
    if args.table_shape is not None:
        cli_config["table_shape"] = args.table_shape
    if args.station_char_shape is not None:
        cli_config["station_char_shape"] = args.station_char_shape
    if args.number_char_shape is not None:
        cli_config["number_char_shape"] = args.number_char_shape
    if args.roi_padding is not None:
        cli_config["roi_padding"] = args.roi_padding
    if args.nchars_threshold is not None:
        cli_config["nchars_threshold"] = args.nchars_threshold
    if args.extra_width is not None:
        cli_config["extra_width"] = args.extra_width
    if args.ocr_engines is not None:
        cli_config["ocr_engines"] = str(args.ocr_engines).split(",")
    cli_config["multi_row_station_names"] = bool(args.multi_row_station_names)
    cli_config["from_extracted_structure"] = bool(args.from_extracted_structure)

    image_formats = (
        str(args.image_formats).split(",")
        if args.image_formats
        else ["png", "jpg", "jpeg", "gif"]
    )

    if debug_dir:
        Path(debug_dir).mkdir(exist_ok=True, parents=True)
        logging.basicConfig(
            level=logging.DEBUG,
            filename=Path(debug_dir) / "reader.log",
            filemode="w",
            format="[%(asctime)s][%(levelname)s]%(name)s - %(message)s",
        )
    else:
        logging.basicConfig(level=logging.ERROR)

    # Validate images directory
    if not images_dir.exists():
        logging.error(f"Error: Images directory not found: {images_dir}")
        sys.exit(1)

    logging.info(f"engines: {DEFAULT_CONFIG.get('ocr_engines', [])}")

    # Configuration discovery
    subtree_configs = discover_configurations(images_dir)

    # Image discovery
    all_image_paths = sorted(
        list(
            chain.from_iterable(
                map(
                    lambda iformat: discover_files(images_dir, f"*.{iformat}"),
                    image_formats,
                )
            )
        )
    )

    images_by_subtree: dict[Path | None, list[Path]] = {}

    # Sort subtrees by depth descending, so deeper configs match first
    sorted_subtrees = sorted(
        subtree_configs.keys(), key=lambda p: len(p.parts), reverse=True
    )

    for img_path in all_image_paths:
        applicable_subtree = None
        for subtree in sorted_subtrees:
            if img_path.is_relative_to(subtree):
                applicable_subtree = subtree
                break

        images_by_subtree.setdefault(applicable_subtree, []).append(img_path)

    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    scan_configs = {}
    group_configs = {}

    for subtree, img_paths in images_by_subtree.items():
        base_config = dict(DEFAULT_CONFIG)
        if subtree and subtree in subtree_configs:
            base_config.update(subtree_configs[subtree])

        merged_config = dict(base_config)
        merged_config.update(cli_config)

        if "ocr_engines" in merged_config and isinstance(
            merged_config["ocr_engines"], str
        ):
            merged_config["ocr_engines"] = merged_config["ocr_engines"].split(",")

        group_configs[subtree] = merged_config

        for img_path in img_paths:
            key = (
                str(img_path.relative_to(images_dir))
                if img_path.is_relative_to(images_dir)
                else str(img_path)
            )
            scan_configs[key] = merged_config

    config_debug_file = output_dir_path / "scans_configurations.json"
    try:
        with open(config_debug_file, "w") as f:
            json.dump(scan_configs, f, indent=4)
        logging.info(f"Configuration debug info written to: {config_debug_file}")
    except Exception as e:
        logging.error(
            f"Failed to write configuration debug info to {config_debug_file}: {e}"
        )

    # Process images partitioned by configuration
    for subtree, img_paths in images_by_subtree.items():
        merged_config = group_configs[subtree]

        group_name = (
            subtree.relative_to(images_dir)
            if subtree and subtree.is_relative_to(images_dir)
            else subtree.name
            if subtree
            else "DEFAULT"
        )
        logging.info(f"\n--- Processing group: {group_name} ---")

        digitize_scans_batch(
            images_dir=images_dir,
            image_paths=img_paths,
            output_dir=output_dir_path,
            debug_dir=Path(debug_dir) if debug_dir else None,
            overwrite=overwrite,
            config=merged_config,
        )


def create_argparser() -> argparse.ArgumentParser:
    """Create and configure argument parser for batch processing."""
    parser = argparse.ArgumentParser(
        prog="nivo-reader",
        description="""Batch digitization of NIVO table images.""",
    )

    # Input/Output arguments
    _ = parser.add_argument(
        "images_dir",
        type=Path,
        help="Directory containing input images",
    )
    _ = parser.add_argument(
        "-o",
        "--output-dir",
        required=True,
        type=str,
        help="Output directory for Excel files",
    )
    _ = parser.add_argument(
        "-d",
        "--debug-dir",
        type=str,
        default=None,
        help="Base directory for debug artifacts (optional)",
    )

    # Table parameters
    _ = parser.add_argument(
        "--clips",
        type=int,
        nargs=4,
        required=False,
        metavar=("UP", "DOWN", "LEFT", "RIGHT"),
        help="Clipping margins for table (up, down, left, right)",
    )
    _ = parser.add_argument(
        "--table-shape",
        type=int,
        nargs=2,
        required=False,
        metavar=("WIDTH", "HEIGHT"),
        help="Expected table shape (width, height)",
    )

    # Character shape parameters
    _ = parser.add_argument(
        "--station-char-shape",
        type=int,
        nargs=2,
        default=None,
        metavar=("WIDTH", "HEIGHT"),
        help="Station name character shape (default: 12 10)",
    )
    _ = parser.add_argument(
        "--number-char-shape",
        type=int,
        nargs=2,
        default=None,
        metavar=("WIDTH", "HEIGHT"),
        help="Number character shape (default: 12 20)",
    )

    # Processing parameters
    _ = parser.add_argument(
        "--roi-padding",
        type=int,
        default=None,
        help="ROI padding in pixels (default: 3)",
    )
    _ = parser.add_argument(
        "--nchars-threshold",
        type=int,
        default=None,
        help="Character count threshold (default: 30)",
    )
    _ = parser.add_argument(
        "--extra-width",
        type=int,
        default=None,
        help="Extra width for cell ROIs (default: 6)",
    )

    # Overwrite flag
    _ = parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files",
    )

    # Image formats
    _ = parser.add_argument(
        "--image-formats",
        type=str,
        default=None,
        help="Comma-separated list of image file formats to process (default: png,jpg,jpeg,gif)",
    )

    # OCR
    _ = parser.add_argument(
        "--ocr-engines",
        type=str,
        default=None,
        help="Comma-separated list of OCR engines to use. Available engines: tesseract, easyocr, paddleocr. Default: tesseract,easyocr,paddleocr",
    )
    _ = parser.add_argument(
        "--multi-row-station-names",
        action="store_true",
        help="Whether station names can span multiple rows",
    )
    _ = parser.add_argument(
        "--from-extracted-structure",
        action="store_true",
        help="Whether table detection should be performed after content cleanup",
    )
    _ = parser.add_argument(
        "--connection", type=str, required=False, help="SqlAlchemy uri to a database"
    )
    _ = parser.add_argument(
        "--run", type=str, required=False, help="Label for the current run"
    )
    return parser


if __name__ == "__main__":
    main()
