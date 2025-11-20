#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Download additional open-source fonts that are not packaged in Debian.

The downloaded fonts are stored under /K3D/Knowledge3D.local/fonts/external/<family>
so they can be referenced by the procedural glyph harvesters.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import logging
import os
import shutil
import sys
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List
from urllib.request import urlopen

EXTERNAL_FONT_ROOT = Path("/K3D/Knowledge3D.local/fonts/external")


@dataclass(frozen=True)
class FontAsset:
    url: str
    filename: str | None = None  # optional rename when saving


@dataclass(frozen=True)
class ExternalFont:
    """Descriptor for a downloadable open font family."""

    name: str
    license: str
    assets: List[FontAsset]


EXTERNAL_FONTS: List[ExternalFont] = [
    ExternalFont(
        name="Atkinson-Hyperlegible",
        license="OFL-1.1",
        assets=[
            FontAsset(
                url="https://raw.githubusercontent.com/google/fonts/main/ofl/atkinsonhyperlegible/AtkinsonHyperlegible-Regular.ttf",
                filename="AtkinsonHyperlegible-Regular.ttf",
            ),
            FontAsset(
                url="https://raw.githubusercontent.com/google/fonts/main/ofl/atkinsonhyperlegible/AtkinsonHyperlegible-Bold.ttf",
                filename="AtkinsonHyperlegible-Bold.ttf",
            ),
        ],
    ),
    ExternalFont(
        name="NotoCJKSans",
        license="OFL-1.1",
        assets=[
            FontAsset(
                url="https://raw.githubusercontent.com/notofonts/noto-cjk/main/Sans/OTF/SimplifiedChinese/NotoSansCJKsc-Regular.otf",
                filename="NotoSansCJKsc-Regular.otf",
            ),
            FontAsset(
                url="https://raw.githubusercontent.com/notofonts/noto-cjk/main/Sans/OTF/TraditionalChinese/NotoSansCJKtc-Regular.otf",
                filename="NotoSansCJKtc-Regular.otf",
            ),
            FontAsset(
                url="https://raw.githubusercontent.com/notofonts/noto-cjk/main/Sans/OTF/Japanese/NotoSansCJKjp-Regular.otf",
                filename="NotoSansCJKjp-Regular.otf",
            ),
            FontAsset(
                url="https://raw.githubusercontent.com/notofonts/noto-cjk/main/Sans/OTF/Korean/NotoSansCJKkr-Regular.otf",
                filename="NotoSansCJKkr-Regular.otf",
            ),
        ],
    ),
    ExternalFont(
        name="ADLaM-Display",
        license="OFL-1.1",
        assets=[
            FontAsset(
                url="https://raw.githubusercontent.com/google/fonts/main/ofl/adlamdisplay/ADLaMDisplay-Regular.ttf",
                filename="ADLaMDisplay-Regular.ttf",
            ),
        ],
    ),
    ExternalFont(
        name="NotoSansCherokee",
        license="OFL-1.1",
        assets=[
            FontAsset(
                url="https://raw.githubusercontent.com/google/fonts/main/ofl/notosanscherokee/NotoSansCherokee%5Bwght%5D.ttf",
                filename="NotoSansCherokee-wght.ttf",
            ),
        ],
    ),
    ExternalFont(
        name="NotoSansCanadianAboriginal",
        license="OFL-1.1",
        assets=[
            FontAsset(
                url="https://raw.githubusercontent.com/google/fonts/main/ofl/notosanscanadianaboriginal/NotoSansCanadianAboriginal%5Bwght%5D.ttf",
                filename="NotoSansCanadianAboriginal-wght.ttf",
            ),
        ],
    ),
]


def _download_bytes(url: str) -> bytes:
    with urlopen(url) as response:  # type: ignore[call-arg]
        return response.read()


def _write_file(target: Path, data: bytes) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("wb") as handle:
        handle.write(data)


def _extract_zip(data: bytes, target_dir: Path, overwrite: bool) -> List[Path]:
    target_dir.mkdir(parents=True, exist_ok=True)
    extracted: List[Path] = []
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        for info in archive.infolist():
            filename = info.filename
            if not filename.lower().endswith((".ttf", ".otf")):
                continue
            relative = Path(filename).name
            destination = target_dir / relative
            if destination.exists() and not overwrite:
                continue
            with archive.open(info) as source, destination.open("wb") as dest:
                shutil.copyfileobj(source, dest)
            extracted.append(destination)
    return extracted


def _save_asset(font: ExternalFont, asset: FontAsset, target_root: Path, overwrite: bool) -> List[Path]:
    logging.info("Downloading %s asset %s", font.name, asset.url)
    data = _download_bytes(asset.url)
    checksum = hashlib.sha256(data).hexdigest()[:12]
    logging.debug("  sha256=%s", checksum)

    target_dir = target_root / font.name
    target_dir.mkdir(parents=True, exist_ok=True)

    if asset.url.lower().endswith(".zip"):
        return _extract_zip(data, target_dir, overwrite=overwrite)

    filename = asset.filename or os.path.basename(asset.url)
    destination = target_dir / filename
    if destination.exists() and not overwrite:
        logging.info("  Skipping existing %s", destination)
        return []
    _write_file(destination, data)
    return [destination]


def download_fonts(selected: Iterable[str], overwrite: bool) -> None:
    target_root = EXTERNAL_FONT_ROOT
    target_root.mkdir(parents=True, exist_ok=True)
    selected_set = {name.lower() for name in selected}
    matched = [font for font in EXTERNAL_FONTS if not selected_set or font.name.lower() in selected_set]
    if not matched:
        logging.warning("No matching fonts for selection: %s", ", ".join(selected))
        return

    for font in matched:
        logging.info("Fetching %s (license: %s)", font.name, font.license)
        for asset in font.assets:
            saved = _save_asset(font, asset, target_root, overwrite=overwrite)
            if saved:
                for path in saved:
                    logging.info("  ↳ %s", path)


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download open-source fonts into .local/fonts/external.")
    parser.add_argument(
        "--fonts",
        nargs="*",
        default=[],
        help="Subset of font families to download (default: all).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download fonts even if files already exist.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available external font families and exit.",
    )
    return parser.parse_args(argv)


def main(argv: List[str] | None = None) -> None:
    args = parse_args(argv or sys.argv[1:])
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    if args.list:
        print("Available external font families:")
        for font in EXTERNAL_FONTS:
            print(f" - {font.name} ({font.license})")
        return

    download_fonts(args.fonts, overwrite=args.force)


if __name__ == "__main__":
    main()
