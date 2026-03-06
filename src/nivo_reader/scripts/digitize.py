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

import easyocr
import numpy as np
import paddleocr
import polars as pl
from PIL import Image
from tqdm import tqdm

from nivo_reader.nivo_reader import read_nivo_table
from nivo_reader.scripts.utils.paths import discover_files

# Suppress pin_memory warnings from PyTorch/EasyOCR
warnings.filterwarnings("ignore", message=".*pin_memory.*")


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
        image = np.array(Image.open(image_path).convert("RGB"))[:, :, ::-1].copy()
        return image
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
    station_char_shape: tuple[int, int] = (12, 10),
    number_char_shape: tuple[int, int] = (12, 20),
    roi_padding: int = 3,
    nchars_threshold: int = 20,
    extra_width: int = 6,
    debug_base_dir: Path | None = None,
    base_images_dir: Path | None = None,
) -> dict[Path, pl.DataFrame]:
    """Process multiple NIVO images and return their digitizations.

    Args:
        image_paths: List of paths to input images
        clips: (up, down, left, right) clipping margins
        table_shape: (width, height) of table
        ocr_engines: List of OCR engines to use ('tesseract', 'easyocr', 'paddleocr')
        station_char_shape: Character dimensions for station names
        number_char_shape: Character dimensions for numbers
        roi_padding: ROI padding in pixels
        nchars_threshold: Character count threshold
        extra_width: Extra width for ROIs
        debug_base_dir: Base directory for debug artifacts
        base_images_dir: Base directory to compute relative paths for debug

    Returns:
        Dictionary mapping input image paths to their digitization DataFrames
    """
    ocrs = {}
    if "easyocr" in ocr_engines:
        ocrs["easyocr"] = easyocr.Reader(lang_list=["it"])
    if "paddleocr" in ocr_engines:
        ocrs["paddleocr"] = paddleocr.TextRecognition(
            model_name="latin_PP-OCRv5_mobile_rec"
        )
    if "tesseract" in ocr_engines:
        ocrs["tesseract"] = None

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
    image_formats: list[str],
    output_dir: Path,
    debug_dir: Path | None,
    clips: tuple[int, int, int, int],
    table_shape: tuple[int, int],
    ocr_engines: list[str],
    station_char_shape: tuple[int, int] = (12, 10),
    number_char_shape: tuple[int, int] = (12, 20),
    roi_padding: int = 3,
    nchars_threshold: int = 20,
    extra_width: int = 6,
    overwrite: bool = False,
) -> None:
    """Batch process multiple NIVO images and write to output files.

    Args:
        images_dir: Directory containing input images
        image_formats: List of image file extensions to process
        output_dir: Directory for output files
        debug_dir: Base directory for debug artifacts
        clips: (up, down, left, right) clipping margins
        table_shape: (width, height) of table
        ocr_engines: List of OCR engines to use
        station_char_shape: Character dimensions for station names
        number_char_shape: Character dimensions for numbers
        roi_padding: ROI padding in pixels
        nchars_threshold: Character count threshold
        extra_width: Extra width for ROIs
        overwrite: Whether to overwrite existing files
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if debug_dir:
        debug_dir = Path(debug_dir)
        debug_dir.mkdir(parents=True, exist_ok=True)

    images_dir = Path(images_dir)
    image_paths = list(
        chain.from_iterable(
            map(lambda format: discover_files(images_dir, format), image_formats)
        )
    )

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
        print("No images to process")
        return

    print("Initializing OCR reader...")

    # We maintain tqdm logic here for TUI functionality.
    # To use our shiny new pure function, we call it in chunks of 1
    # so we still get proper per-image TUI progress bars.
    print(f"Processing {len(images_to_process)} images...")

    for img_path in tqdm(
        images_to_process, desc="Processing images"
    ):  # pyrefly: ignore
        results = extract_digitizations(
            image_paths=[img_path],
            clips=clips,
            table_shape=table_shape,
            ocr_engines=ocr_engines,
            station_char_shape=station_char_shape,
            number_char_shape=number_char_shape,
            roi_padding=roi_padding,
            nchars_threshold=nchars_threshold,
            extra_width=extra_width,
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


def create_parser() -> argparse.ArgumentParser:
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
        required=True,
        metavar=("UP", "DOWN", "LEFT", "RIGHT"),
        help="Clipping margins for table (up, down, left, right)",
    )
    _ = parser.add_argument(
        "--table-shape",
        type=int,
        nargs=2,
        required=True,
        metavar=("WIDTH", "HEIGHT"),
        help="Expected table shape (width, height)",
    )

    # Character shape parameters
    _ = parser.add_argument(
        "--station-char-shape",
        type=int,
        nargs=2,
        default=[12, 10],
        metavar=("WIDTH", "HEIGHT"),
        help="Station name character shape (default: 12 10)",
    )
    _ = parser.add_argument(
        "--number-char-shape",
        type=int,
        nargs=2,
        default=[12, 20],
        metavar=("WIDTH", "HEIGHT"),
        help="Number character shape (default: 12 20)",
    )

    # Processing parameters
    _ = parser.add_argument(
        "--roi-padding",
        type=int,
        default=3,
        help="ROI padding in pixels (default: 3)",
    )
    _ = parser.add_argument(
        "--nchars-threshold",
        type=int,
        default=30,
        help="Character count threshold (default: 30)",
    )
    _ = parser.add_argument(
        "--extra-width",
        type=int,
        default=6,
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
        default="png,jpg,jpeg,gif",
        help="Comma-separated list of image file formats to process (default: png,jpg,jpeg,gif)",
    )

    # OCR
    _ = parser.add_argument(
        "--ocr-engines",
        type=str,
        default="tesseract,easyocr,paddleocr",
        help="Comma-separated list of OCR engines to use. Available engines: tesseract, easyocr, paddleocr. Default: tesseract,easyocr,paddleocr",
    )
    return parser


def main():
    parser = create_parser()
    args = parser.parse_args()

    # Create variables explicitly to satisfy basedpyright
    debug_dir: str | None = args.debug_dir
    images_dir: Path = args.images_dir
    output_dir: str = args.output_dir
    clips: list[int] = args.clips
    table_shape: list[int] = args.table_shape
    station_char_shape: list[int] = args.station_char_shape
    number_char_shape: list[int] = args.number_char_shape
    roi_padding: int = args.roi_padding
    nchars_threshold: int = args.nchars_threshold
    extra_width: int = args.extra_width
    overwrite: bool = args.overwrite
    image_formats: list[str] = str(args.image_formats).split(",")
    ocr_engines: list[str] = str(args.ocr_engines).split(",")

    if debug_dir:
        Path(debug_dir).mkdir(exist_ok=True, parents=True)
        logging.basicConfig(
            level=logging.DEBUG,
            filename=Path(debug_dir) / "process_nivo_images.log",
            filemode="w",
        )
    else:
        logging.basicConfig(level=logging.CRITICAL)

    # Validate images directory
    if not images_dir.exists():
        print(f"Error: Images directory not found: {images_dir}")
        sys.exit(1)

    logging.info(f"engines: {ocr_engines}")

    # Process images
    digitize_scans_batch(
        images_dir=images_dir,
        image_formats=image_formats,
        output_dir=Path(output_dir),
        debug_dir=Path(debug_dir) if debug_dir else None,
        clips=(clips[0], clips[1], clips[2], clips[3]),
        table_shape=(table_shape[0], table_shape[1]),
        station_char_shape=(station_char_shape[0], station_char_shape[1]),
        number_char_shape=(number_char_shape[0], number_char_shape[1]),
        roi_padding=roi_padding,
        nchars_threshold=nchars_threshold,
        extra_width=extra_width,
        overwrite=overwrite,
        ocr_engines=ocr_engines,
    )


if __name__ == "__main__":
    main()
