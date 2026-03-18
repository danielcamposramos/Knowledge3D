"""Standard paper and book trim sizes."""

from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class StandardSize:
    key: str
    label: str
    width_mm: float
    height_mm: float
    standard: str
    category: str


PAPER_SIZES: dict[str, tuple[float, float]] = {
    "A0": (841.0, 1189.0),
    "A1": (594.0, 841.0),
    "A2": (420.0, 594.0),
    "A3": (297.0, 420.0),
    "A4": (210.0, 297.0),
    "A5": (148.0, 210.0),
    "A6": (105.0, 148.0),
    "A7": (74.0, 105.0),
    "A8": (52.0, 74.0),
    "A9": (37.0, 52.0),
    "A10": (26.0, 37.0),
    "B0": (1000.0, 1414.0),
    "B1": (707.0, 1000.0),
    "B2": (500.0, 707.0),
    "B3": (353.0, 500.0),
    "B4": (250.0, 353.0),
    "B5": (176.0, 250.0),
    "C0": (917.0, 1297.0),
    "C3": (324.0, 458.0),
    "C4": (229.0, 324.0),
    "C5": (162.0, 229.0),
    "C6": (114.0, 162.0),
    "DL": (110.0, 220.0),
    "Letter": (215.9, 279.4),
    "Legal": (215.9, 355.6),
    "Tabloid": (279.4, 431.8),
    "Ledger": (431.8, 279.4),
    "Executive": (184.2, 266.7),
    "Half_Letter": (139.7, 215.9),
    "JIS_B0": (1030.0, 1456.0),
    "JIS_B1": (728.0, 1030.0),
    "JIS_B2": (515.0, 728.0),
    "JIS_B3": (364.0, 515.0),
    "JIS_B4": (257.0, 364.0),
    "JIS_B5": (182.0, 257.0),
}

BOOK_SIZES: dict[str, tuple[float, float]] = {
    "mass_market_paperback": (108.0, 175.0),
    "trade_paperback_small": (127.0, 203.0),
    "trade_paperback": (140.0, 216.0),
    "trade_paperback_large": (152.0, 229.0),
    "hardcover_small": (140.0, 216.0),
    "hardcover_standard": (152.0, 229.0),
    "hardcover_large": (178.0, 254.0),
    "textbook": (203.0, 254.0),
    "coffee_table": (254.0, 305.0),
    "quarto": (190.0, 250.0),
    "octavo": (152.0, 229.0),
    "folio": (305.0, 483.0),
    "pamphlet": (140.0, 216.0),
    "leaflet": (99.0, 210.0),
    "broadsheet": (375.0, 600.0),
    "tabloid_newspaper": (280.0, 430.0),
}


def iter_paper_sizes() -> list[StandardSize]:
    out: list[StandardSize] = []
    for key, (width, height) in PAPER_SIZES.items():
        standard = "ISO_216"
        if key in {"Letter", "Legal", "Tabloid", "Ledger", "Executive", "Half_Letter"}:
            standard = "North_American"
        elif key.startswith("JIS_"):
            standard = "JIS"
        out.append(StandardSize(key=key, label=key.replace("_", " "), width_mm=width, height_mm=height, standard=standard, category="paper"))
    return out


def iter_book_sizes() -> list[StandardSize]:
    return [
        StandardSize(key=key, label=key.replace("_", " "), width_mm=width, height_mm=height, standard="Book_Trim", category="book")
        for key, (width, height) in sorted(BOOK_SIZES.items())
    ]


def a_series_ratio_ok(tolerance_mm: float = 1.0) -> bool:
    series = ["A0", "A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8", "A9", "A10"]
    for prev_key, next_key in zip(series, series[1:]):
        prev_width, prev_height = PAPER_SIZES[prev_key]
        next_width, next_height = PAPER_SIZES[next_key]
        if abs((prev_height / 2.0) - next_width) > tolerance_mm:
            return False
        if abs(prev_width - next_height) > tolerance_mm:
            return False
        ratio = max(prev_width, prev_height) / min(prev_width, prev_height)
        if abs(ratio - math.sqrt(2.0)) > 0.02:
            return False
    return True


__all__ = ["BOOK_SIZES", "PAPER_SIZES", "StandardSize", "a_series_ratio_ok", "iter_book_sizes", "iter_paper_sizes"]
