#!/usr/bin/env python3
import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm
import easyocr
import polars as pl
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


def process_images(
    image_paths: list[Path],
    output_dir: Path,
    debug_dir: Path,
    clips: tuple[int, int, int, int],
    table_shape: tuple[int, int],
    anagrafica: list[str],
    station_char_shape: tuple[int, int] = (12, 10),
    number_char_shape: tuple[int, int] = (12, 20),
    roi_padding: int = 3,
    nchars_threshold: int = 30,
    extra_width: int = 6,
    low_confidence_threshold: float = 0.7,
    overwrite: bool = False,
) -> None:
    """Process multiple NIVO images and write to Excel files.

    Args:
        image_paths: list of image file paths
        output_dir: Directory for output Excel files
        debug_dir: Base directory for debug artifacts
        clips: (up, down, left, right) clipping margins
        table_shape: (width, height) of table
        anagrafica: list of known station names
        station_char_shape: Character dimensions for station names
        number_char_shape: Character dimensions for numbers
        roi_padding: ROI padding in pixels
        nchars_threshold: Character count threshold
        extra_width: Extra width for ROIs
        low_confidence_threshold: Confidence threshold for output
        overwrite: Whether to overwrite existing files
    """
    # Initialize OCR reader once
    print("Initializing OCR reader...")
    ocr = easyocr.Reader(lang_list=["it"])

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    debug_base = Path(debug_dir) if debug_dir else None
    if debug_base:
        debug_base.mkdir(parents=True, exist_ok=True)

    # Filter images to process
    images_to_process = []
    for img_path in image_paths:
        img_path = Path(img_path)
        excel_out = output_dir / img_path.stem / ".xlsx"

        # Check if already processed
        if excel_out.exists() and not overwrite:
            print(f"Skipping {img_path.name} (already processed)")
            continue

        images_to_process.append(img_path)

    if not images_to_process:
        print("No images to process")
        return

    print(f"Processing {len(images_to_process)} images...")

    # Process images with progress bar
    for img_path in tqdm(images_to_process, desc="Processing images"):
        try:
            # Load image
            image = load_image(img_path)

            # Generate output paths
            excel_out_path = output_dir / f"{img_path.stem}.xlsx"

            # Generate debug directory if requested
            img_debug_dir = None
            if debug_base:
                img_debug_dir = debug_base / img_path.stem

            # Process image
            read_nivo_table(
                image,
                excel_out_path,
                ocr,
                clips=clips,
                table_shape=table_shape,
                anagrafica=anagrafica,
                station_char_shape=station_char_shape,
                number_char_shape=number_char_shape,
                roi_padding=roi_padding,
                nchars_threshold=nchars_threshold,
                extra_width=extra_width,
                low_confidence_threshold=low_confidence_threshold,
                debug_dir=img_debug_dir,
            )

            tqdm.write(f"✓ Processed: {img_path.name}")

        except Exception as e:
            tqdm.write(f"✗ Error processing {img_path.name}: {e}")


def create_argparser():
    """Parse arguments and execute batch processing."""
    parser = argparse.ArgumentParser(
        description="Batch process NIVO table images and extract to Excel"
    )

    # Input/Output arguments
    parser.add_argument(
        "image_files",
        nargs="+",
        type=str,
        help="Input image file paths",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        required=True,
        type=str,
        help="Output directory for Excel files",
    )
    parser.add_argument(
        "-d",
        "--debug-dir",
        type=str,
        default=None,
        help="Base directory for debug artifacts (optional)",
    )

    # Table parameters
    parser.add_argument(
        "--clips",
        type=int,
        nargs=4,
        required=True,
        metavar=("UP", "DOWN", "LEFT", "RIGHT"),
        help="Clipping margins for table (up, down, left, right)",
    )
    parser.add_argument(
        "--table-shape",
        type=int,
        nargs=2,
        required=True,
        metavar=("WIDTH", "HEIGHT"),
        help="Expected table shape (width, height)",
    )
    parser.add_argument(
        "--anagrafica-file",
        required=True,
        type=str,
        help="File with station names (one per line)",
    )

    # Character shape parameters
    parser.add_argument(
        "--station-char-shape",
        type=int,
        nargs=2,
        default=[12, 10],
        metavar=("WIDTH", "HEIGHT"),
        help="Station name character shape (default: 12 10)",
    )
    parser.add_argument(
        "--number-char-shape",
        type=int,
        nargs=2,
        default=[12, 20],
        metavar=("WIDTH", "HEIGHT"),
        help="Number character shape (default: 12 20)",
    )

    # Processing parameters
    parser.add_argument(
        "--roi-padding",
        type=int,
        default=3,
        help="ROI padding in pixels (default: 3)",
    )
    parser.add_argument(
        "--nchars-threshold",
        type=int,
        default=30,
        help="Character count threshold (default: 30)",
    )
    parser.add_argument(
        "--extra-width",
        type=int,
        default=6,
        help="Extra width for cell ROIs (default: 6)",
    )
    parser.add_argument(
        "--low-confidence-threshold",
        type=float,
        default=0.7,
        help="Confidence threshold for output (default: 0.7)",
    )

    # Overwrite flag
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files",
    )

    return parser


def main():
    parser = create_argparser()
    args = parser.parse_args()

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

    # Convert image paths
    image_paths = [Path(img) for img in args.image_files]

    # Validate image paths
    for img_path in image_paths:
        if not img_path.exists():
            print(f"Error: Image file not found: {img_path}")
            sys.exit(1)

    # Process images
    process_images(
        image_paths,
        output_dir=args.output_dir,
        debug_dir=args.debug_dir,
        clips=tuple(args.clips),
        table_shape=tuple(args.table_shape),
        anagrafica=anagrafica,
        station_char_shape=tuple(args.station_char_shape),
        number_char_shape=tuple(args.number_char_shape),
        roi_padding=args.roi_padding,
        nchars_threshold=args.nchars_threshold,
        extra_width=args.extra_width,
        low_confidence_threshold=args.low_confidence_threshold,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
