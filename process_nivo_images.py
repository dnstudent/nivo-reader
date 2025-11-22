#!/usr/bin/env python3
import argparse
import sys
import warnings
from itertools import chain
from pathlib import Path
import logging

import easyocr
import numpy as np
import paddleocr
import polars as pl
from PIL import Image
from tqdm import tqdm

from nivo_reader import read_nivo_table

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


def excel_out(output_dir: Path, image_path: Path, images_dir: Path) -> Path:
    """Generate output Excel file path for an image.

    Args:
        output_dir: Output directory for Excel files
        image_path: Path to the input image
        images_dir: Base directory containing images

    Returns:
        Path to the output Excel file
    """
    return output_dir / Path(str(image_path.relative_to(images_dir)) + ".xlsx")


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


def process_images(
    images_dir: Path,
    image_formats: list[str],
    output_dir: Path,
    debug_dir: Path,
    clips: tuple[int, int, int, int],
    table_shape: tuple[int, int],
    anagrafica: list[str],
    station_char_shape: tuple[int, int] = (12, 10),
    number_char_shape: tuple[int, int] = (12, 20),
    roi_padding: int = 3,
    nchars_threshold: int = 20,
    extra_width: int = 6,
    low_confidence_threshold: float = 0.7,
    overwrite: bool = False,
) -> None:
    """Process multiple NIVO images and write to Excel files.

    Args:
        images_dir: Directory containing input images
        image_formats: List of image file extensions to process (e.g., ['png', 'jpg'])
        output_dir: Directory for output Excel files
        debug_dir: Base directory for debug artifacts
        clips: (up, down, left, right) clipping margins
        table_shape: (width, height) of table
        anagrafica: List of known station names
        station_char_shape: Character dimensions for station names
        number_char_shape: Character dimensions for numbers
        roi_padding: ROI padding in pixels
        nchars_threshold: Character count threshold
        extra_width: Extra width for ROIs
        low_confidence_threshold: Confidence threshold for output
        overwrite: Whether to overwrite existing files
    """
    # Initialize OCR reader once

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    debug_base = Path(debug_dir) if debug_dir else None
    if debug_base:
        debug_base.mkdir(parents=True, exist_ok=True)

    images_dir = Path(images_dir)
    image_paths = list(
        chain.from_iterable(
            map(lambda format: images_dir.glob(f"**/*.{format}"), image_formats)
        )
    )

    # Filter images to process
    if not overwrite:
        images_to_process = [
            image_path
            for image_path in image_paths
            if not excel_out(output_dir, image_path, images_dir).exists()
        ]
    else:
        images_to_process = image_paths

    if not images_to_process:
        print("No images to process")
        return

    print("Initializing OCR reader...")

    easyreader = easyocr.Reader(lang_list=["it"])
    paddletextrecog = None #paddleocr.TextRecognition(model_name="latin_PP-OCRv5_mobile_rec")

    print(f"Processing {len(images_to_process)} images...")
    # Process images with progress bar
    for img_path in tqdm(images_to_process, desc="Processing images"):  # pyrefly: ignore
        try:
            # Load image
            image = load_image(img_path)

            # Generate output paths
            excel_out_path = excel_out(output_dir, img_path, images_dir)

            # Generate debug directory if requested
            img_debug_dir = None
            if debug_base:
                img_debug_dir = compose_debug_dir(debug_base, img_path, images_dir)

            # Process image
            read_nivo_table(
                image,
                excel_out_path,
                clips,
                table_shape,
                anagrafica,
                easyreader,
                paddletextrecog,
                station_char_shape,
                number_char_shape,
                roi_padding,
                nchars_threshold,
                extra_width,
                low_confidence_threshold,
                img_debug_dir,
            )

            tqdm.write(f"✓ Processed: {img_path.relative_to(images_dir)}")

        except Exception as e:
            logging.exception(
                f"Error processing {img_path.relative_to(images_dir)}: {e}"
            )
            tqdm.write(f"✗ Error processing {img_path.relative_to(images_dir)}: {e}")


def create_argparser() -> argparse.ArgumentParser:
    """Create and configure argument parser for batch processing."""
    parser = argparse.ArgumentParser(
        description="Batch process NIVO table images and extract to Excel"
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
    _ = parser.add_argument(
        "--anagrafica-file",
        required=True,
        type=str,
        help="File with station names (one per line)",
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
    _ = parser.add_argument(
        "--low-confidence-threshold",
        type=float,
        default=0.7,
        help="Confidence threshold for output (default: 0.7)",
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

    return parser


def main():
    parser = create_argparser()
    args = parser.parse_args()

    if args.debug_dir:
        Path(args.debug_dir).mkdir(exist_ok=True, parents=True)
        logging.basicConfig(
            level=logging.DEBUG,
            filename=Path(args.debug_dir) / "process_nivo_images.log",
            filemode="w",
        )
    else:
        logging.basicConfig(level=logging.CRITICAL)

    # Load station names
    anagrafica_path = Path(args.anagrafica_file)
    if not anagrafica_path.exists():
        print(f"Error: Anagrafica file not found: {anagrafica_path}")
        sys.exit(1)

    anagrafica = (
        pl.read_excel(
            args.anagrafica_file,
            columns=["Stazione"],
        )
        .filter(pl.col("Stazione").is_not_null())["Stazione"]
        .to_list()
    )

    if not anagrafica:
        print("Error: Anagrafica file is empty")
        sys.exit(1)

    # Validate images directory
    if not args.images_dir.exists():
        print(f"Error: Images directory not found: {args.images_dir}")
        sys.exit(1)

    image_formats = str(args.image_formats).split(",")

    # Process images
    process_images(
        images_dir=args.images_dir,
        image_formats=image_formats,
        output_dir=args.output_dir,
        debug_dir=args.debug_dir,
        clips=(args.clips[0], args.clips[1], args.clips[2], args.clips[3]),
        table_shape=(args.table_shape[0], args.table_shape[1]),
        anagrafica=anagrafica,
        station_char_shape=(args.station_char_shape[0], args.station_char_shape[1]),
        number_char_shape=(args.number_char_shape[0], args.number_char_shape[1]),
        roi_padding=args.roi_padding,
        nchars_threshold=args.nchars_threshold,
        extra_width=args.extra_width,
        low_confidence_threshold=args.low_confidence_threshold,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
