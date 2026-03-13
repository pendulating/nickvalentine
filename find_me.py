#!/usr/bin/env python3
"""
find_me.py - Find photos of yourself in a folder of event photos.

Uses face_recognition to compare faces against one or more reference photos,
then copies matches to an output folder. Supports JPG, PNG, RAW (RAF, CR2,
CR3, NEF, ARW, DNG), and HEIC files. Scans subdirectories recursively.
"""

import argparse
import os
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import face_recognition
import numpy as np
import rawpy
from PIL import Image
from tqdm import tqdm

STANDARD_EXTS = {".png", ".jpg", ".jpeg"}
RAW_EXTS = {".raf", ".cr2", ".cr3", ".nef", ".arw", ".dng"}
HEIC_EXTS = {".heic", ".heif"}
ALL_EXTS = STANDARD_EXTS | RAW_EXTS | HEIC_EXTS


def load_image(file_path: str, max_dimension: int = 1600) -> np.ndarray:
    """Load an image (standard, RAW, or HEIC), resize it, return as numpy RGB array."""
    ext = Path(file_path).suffix.lower()

    if ext in RAW_EXTS:
        with rawpy.imread(file_path) as raw:
            rgb = raw.postprocess(half_size=True)
            img = Image.fromarray(rgb)
    elif ext in HEIC_EXTS:
        import pillow_heif

        heif_file = pillow_heif.read_heif(file_path)
        img = Image.frombytes(heif_file.mode, heif_file.size, heif_file.data)
        img = img.convert("RGB")
    else:
        img = Image.open(file_path).convert("RGB")

    img.thumbnail((max_dimension, max_dimension), Image.Resampling.LANCZOS)
    return np.array(img)


def encode_reference(ref_path: str) -> np.ndarray | None:
    """Load a reference photo and return its face encoding, or None if no face found."""
    img = load_image(ref_path)
    encodings = face_recognition.face_encodings(img)
    if not encodings:
        print(f"  Warning: No face found in {ref_path}, skipping.", file=sys.stderr)
        return None
    if len(encodings) > 1:
        print(f"  Warning: Multiple faces in {ref_path}, using the first one.", file=sys.stderr)
    return encodings[0]


def check_photo(file_path: str, known_encodings: list[np.ndarray], tolerance: float) -> bool:
    """Check if any face in file_path matches any of the known encodings."""
    img = load_image(file_path, max_dimension=800)
    face_encodings = face_recognition.face_encodings(img)
    for face_enc in face_encodings:
        matches = face_recognition.compare_faces(known_encodings, face_enc, tolerance=tolerance)
        if any(matches):
            return True
    return False


def _init_worker(niceness: int):
    """Initializer for worker processes: lower priority so we don't starve the system."""
    os.nice(niceness)


def _worker(args: tuple) -> tuple[str, bool, str]:
    """Worker function for parallel processing. Returns (path, matched, error)."""
    file_path, known_encodings, tolerance = args
    try:
        matched = check_photo(file_path, known_encodings, tolerance)
        return (file_path, matched, "")
    except Exception as e:
        return (file_path, False, str(e))


def collect_photos(search_dir: str) -> list[str]:
    """Recursively find all supported image files."""
    photos = []
    for root, _dirs, files in os.walk(search_dir):
        for f in files:
            if Path(f).suffix.lower() in ALL_EXTS:
                photos.append(os.path.join(root, f))
    photos.sort()
    return photos


def main():
    parser = argparse.ArgumentParser(
        description="Find photos of yourself in a folder of event photos.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
examples:
  %(prog)s -r ref.jpg ~/Downloads/NYFW_BRIDAL
  %(prog)s -r ref1.jpg -r ref2.jpg ~/Downloads/NYFW_BRIDAL -o ~/Pictures/nyfw_me
  %(prog)s -r ~/Downloads/me/*.jpg ~/Downloads/NYFW_BRIDAL --tolerance 0.5
""",
    )
    parser.add_argument(
        "search_dir",
        help="Folder to scan for photos (scanned recursively).",
    )
    parser.add_argument(
        "-r",
        "--reference",
        required=True,
        action="append",
        help="Reference photo of your face. Pass multiple times for multiple refs (-r a.jpg -r b.jpg).",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Output folder for matched photos (default: <search_dir>/found_me).",
    )
    parser.add_argument(
        "-t",
        "--tolerance",
        type=float,
        default=0.6,
        help="Face match tolerance. Lower = stricter. (default: 0.6)",
    )
    parser.add_argument(
        "-w",
        "--workers",
        type=int,
        default=2,
        help="Number of parallel workers (default: 2).",
    )
    parser.add_argument(
        "--nice",
        type=int,
        default=10,
        help="Process niceness (0-20). Higher = more polite to other apps. (default: 10)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show matches without copying files.",
    )
    parser.add_argument(
        "--flat",
        action="store_true",
        help="Copy all matches into output folder without preserving subfolder structure.",
    )

    args = parser.parse_args()

    search_dir = args.search_dir
    ref_paths = args.reference

    if not os.path.isdir(search_dir):
        parser.error(f"Search directory not found: {search_dir}")
    for rp in ref_paths:
        if not os.path.isfile(rp):
            parser.error(f"Reference photo not found: {rp}")

    output_dir = args.output or os.path.join(search_dir, "found_me")
    os.makedirs(output_dir, exist_ok=True)

    # --- Encode reference faces ---
    print(f"Loading {len(ref_paths)} reference photo(s)...")
    known_encodings = []
    for rp in ref_paths:
        enc = encode_reference(rp)
        if enc is not None:
            known_encodings.append(enc)
            print(f"  Encoded: {rp}")

    if not known_encodings:
        print("Error: No usable face found in any reference photo.", file=sys.stderr)
        sys.exit(1)

    # --- Collect photos ---
    photos = collect_photos(search_dir)
    # Exclude anything already in the output folder
    abs_output = os.path.abspath(output_dir)
    photos = [p for p in photos if not os.path.abspath(p).startswith(abs_output)]

    if not photos:
        print("No supported photos found.")
        return

    print(f"Found {len(photos)} photos to scan.\n")

    # --- Scan ---
    matched_files: list[str] = []
    errors: list[tuple[str, str]] = []

    worker_args = [(p, known_encodings, args.tolerance) for p in photos]

    # Nice the main process too
    os.nice(args.nice)

    with ProcessPoolExecutor(
        max_workers=args.workers,
        initializer=_init_worker,
        initargs=(args.nice,),
    ) as executor:
        futures = {executor.submit(_worker, wa): wa[0] for wa in worker_args}
        with tqdm(total=len(photos), unit="photo", desc="Scanning") as pbar:
            for future in as_completed(futures):
                file_path, matched, error = future.result()
                if error:
                    errors.append((file_path, error))
                elif matched:
                    matched_files.append(file_path)
                pbar.update(1)

    # --- Copy matches ---
    if matched_files:
        matched_files.sort()
        print(f"\nFound {len(matched_files)} match(es)!")
        for fp in matched_files:
            rel = os.path.relpath(fp, search_dir)
            if args.flat:
                dest = os.path.join(output_dir, os.path.basename(fp))
            else:
                dest = os.path.join(output_dir, rel)
                os.makedirs(os.path.dirname(dest), exist_ok=True)

            if args.dry_run:
                print(f"  [dry-run] {rel}")
            else:
                shutil.copy2(fp, dest)
                print(f"  Copied: {rel}")
    else:
        print("\nNo matches found.")

    if errors:
        print(f"\n{len(errors)} file(s) had errors:")
        for fp, err in errors:
            print(f"  {os.path.basename(fp)}: {err}")

    # --- Summary ---
    print(f"\nSummary: {len(matched_files)} matched / {len(photos)} scanned / {len(errors)} errors")
    if matched_files and not args.dry_run:
        print(f"Output:  {output_dir}")


if __name__ == "__main__":
    main()
