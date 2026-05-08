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
from pathlib import Path
from typing import Any

from cv2.typing import MatLike
import numpy as np
import polars as pl
from PIL import Image
from pydantic import BaseModel, DirectoryPath, NonNegativeInt, PositiveInt
from tqdm import tqdm


from nivo_reader.nivo_reader import read_nivo_table
from nivo_reader.scripts.utils.paths import build_config_stack, filter_stack


# Suppress pin_memory warnings from PyTorch/EasyOCR
warnings.filterwarnings("ignore", message=".*pin_memory.*")
logging.getLogger("paddlex").setLevel(logging.ERROR)
logging.getLogger("paddle").setLevel(logging.ERROR)
logging.getLogger("pytesseract").setLevel(logging.ERROR)


class RectSpec(BaseModel):
    width: PositiveInt
    height: PositiveInt


class ClipSpec(BaseModel):
    top: NonNegativeInt
    bottom: NonNegativeInt = 0
    left: NonNegativeInt = 0
    right: NonNegativeInt = 0


class PipelineConfig(BaseModel):
    table_shape: RectSpec
    clips: ClipSpec
    station_char_shape: RectSpec = RectSpec(width=12, height=10)
    number_char_shape: RectSpec = RectSpec(width=12, height=20)
    roi_padding: NonNegativeInt = 3
    nchars_threshold: NonNegativeInt = 20
    extra_width: NonNegativeInt = 6
    multi_row_station_names: bool = False
    from_extracted_structure: bool = False


class AppConfig(BaseModel):
    output_dir: Path
    scans_dir: DirectoryPath
    debug_dir: Path | None = None
    image_formats: set[str] = {"png", "jpg", "jpeg", "gif"}
    overwrite: bool = False
    ocr_engines: set[str] = {"tesseract", "easyocr", "paddleocr"}
    pipeline_config_fname: str = "config.toml"


_OCR_CACHE = {}


def get_ocrs(ocr_engines: set[str]) -> dict[str, Any]:
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


def compose_output_path(
    output_dir: Path, scan_input_path: Path, scans_dir: Path
) -> Path:
    return (
        scan_output_dir(output_dir, scan_input_path, scans_dir)
        / "raw_digitization.json"
    )


def already_digitized(output_dir: Path, path: Path, scans_dir: Path) -> bool:
    return compose_output_path(output_dir, path, scans_dir).exists()


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


def digitize(
    scan: MatLike,
    scan_config: PipelineConfig,
    ocrs: dict[str, Any],
    debug_dir: Path | None,
) -> pl.DataFrame:
    clips = (
        scan_config.clips.top,
        scan_config.clips.bottom,
        scan_config.clips.left,
        scan_config.clips.right,
    )
    table_shape = (
        scan_config.table_shape.width,
        scan_config.table_shape.height,
    )
    station_char_shape = (
        scan_config.station_char_shape.width,
        scan_config.station_char_shape.height,
    )
    number_char_shape = (
        scan_config.number_char_shape.width,
        scan_config.number_char_shape.height,
    )
    return read_nivo_table(
        scan,
        clips,
        table_shape,
        ocrs,
        scan_config.multi_row_station_names,
        scan_config.from_extracted_structure,
        station_char_shape,
        number_char_shape,
        scan_config.roi_padding,
        scan_config.nchars_threshold,
        scan_config.extra_width,
        debug_dir,
    )


def main():
    parser = create_argparser()
    args = parser.parse_args()

    cli_params = {
        k: v
        for k, v in vars(args).items()
        if v is not None and k in AppConfig.model_fields
    }
    script_config = AppConfig(**cli_params)

    scans_dir = script_config.scans_dir
    # Validate images directory
    if not scans_dir.exists():
        logging.error(f"Error: Images directory not found: {scans_dir}")
        sys.exit(1)

    output_dir = script_config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    overwrite = script_config.overwrite
    image_formats = script_config.image_formats

    debug_dir = script_config.debug_dir
    if debug_dir:
        Path(debug_dir).mkdir(exist_ok=True, parents=True)
        logging.basicConfig(
            level=logging.INFO,
            filename=Path(debug_dir) / "reader.log",
            filemode="w",
            format="[%(asctime)s][%(levelname)s]%(name)s - %(message)s",
        )
    else:
        logging.basicConfig(level=logging.ERROR)

    logging.info(f"engines: {script_config.ocr_engines}")

    # Configuration discovery
    scan_config_stack = build_config_stack(
        root=scans_dir,
        model=PipelineConfig,
        config_filename=script_config.pipeline_config_fname,
    )

    def to_digitize(path: Path) -> bool:
        return path.suffix.strip(".") in image_formats and (
            overwrite or not already_digitized(output_dir, path, scans_dir)
        )

    scan_config_stack = filter_stack(scan_config_stack, to_digitize)
    scan_items = sorted(scan_config_stack.items(), key=lambda x: x[0])

    ocrs = get_ocrs(script_config.ocr_engines)

    pbar = tqdm(scan_items, desc="Processing scans")
    for scan_path, scan_config in pbar:
        pbar.set_description(f"Processing {scan_path.name}")
        scan_debug_dir = (
            compose_debug_dir(debug_dir, scan_path, scans_dir) if debug_dir else None
        )
        try:
            scan = load_image(scan_path)
            df = digitize(scan, scan_config, ocrs, scan_debug_dir)
            out = compose_output_path(output_dir, scan_path, scans_dir)
            out.parent.mkdir(parents=True, exist_ok=True)
            df.write_json(out)
            tqdm.write(f"✓ Processed: {scan_path.relative_to(scans_dir)}")
        except Exception as e:
            tqdm.write(f"✗ Error processing {scan_path.relative_to(scans_dir)}: {e}")
            logging.exception(f"Error processing {scan_path}: {e}")


def create_argparser() -> argparse.ArgumentParser:
    """Create and configure argument parser for batch processing."""
    parser = argparse.ArgumentParser(
        prog="nivo-reader",
        description="""Batch digitization of NIVO table images.""",
    )

    # Input/Output arguments
    _ = parser.add_argument(
        "-s",
        "--scans-dir",
        required=True,
        type=Path,
        help="Directory containing input images",
    )
    _ = parser.add_argument(
        "-o",
        "--output-dir",
        required=True,
        type=Path,
        help="Output directory for Excel files",
    )
    _ = parser.add_argument(
        "-d",
        "--debug-dir",
        type=Path,
        help="Base directory for debug artifacts. Optional.",
    )

    # Overwrite flag
    _ = parser.add_argument(
        "-w",
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files",
    )

    # Image formats
    _ = parser.add_argument(
        "--image-formats",
        type=lambda s: set(s.split(",")),
        help=f"Comma-separated list of image file formats to process (default: {','.join(AppConfig.model_fields['image_formats'].default)}",
    )

    # OCR
    _ = parser.add_argument(
        "--ocr-engines",
        type=lambda s: set(s.split(",")),
        help=f"Comma-separated list of OCR engines to use. Available engines: tesseract, tesserocr, easyocr, paddleocr. Default: {','.join(AppConfig.model_fields['ocr_engines'].default)}",
    )

    return parser


if __name__ == "__main__":
    main()
