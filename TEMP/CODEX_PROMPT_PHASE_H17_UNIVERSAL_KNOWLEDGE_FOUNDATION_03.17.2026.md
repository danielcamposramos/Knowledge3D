# Phase H17 — Universal Knowledge Foundation

**Depends on:** H16c (Multi-Modal Provider Layer), Foundational Knowledge Specification
**Creates:** `knowledge3d/ingestion/universal_knowledge/` (package with submodules)
**Tests:** `tests/test_universal_knowledge.py`

---

## Vision

Populate the Galaxy and House with **universal foundational knowledge** that every intelligence needs. This is the content that makes K3D a true knowledge system — not just a shell with navigation, but a brain with actual knowledge in it.

**Daniel's principle:** "The meaning, as it is a concept from an original language, the first description will be of the concept in English because it's the W3C standard worldwide. And the second option to include everyone, including any language from any time, any type of writing."

**Proceduralize = Symlink:** "To transform the content into symlink content. Word by word, you transform it into the symlink of the meaning."

---

## Architecture: Meaning-First Multilingualism

### The Symlink Star Pattern

Every concept in K3D is a **meaning-centric star** — one concept, all languages:

```
Star: concept_water
├── meaning_id: "water"
├── meaning_class: "substance"
├── definition_rpn: "SUBSTANCE LIQUID H2O MOLECULE 2_HYDROGEN 1_OXYGEN"
├── surface_forms:
│   ├── en: "water"         # W3C primary (always present)
│   ├── pt: "água"
│   ├── es: "agua"
│   ├── fr: "eau"
│   ├── de: "Wasser"
│   ├── zh: "水"            # Chinese character
│   ├── ja: "水 / みず"     # Kanji + Hiragana
│   ├── ar: "ماء"           # Arabic
│   ├── hi: "पानी"          # Hindi/Devanagari
│   ├── he: "מים"           # Hebrew
│   ├── ko: "물"            # Korean
│   ├── el: "νερό"          # Greek
│   ├── ru: "вода"          # Cyrillic
│   ├── sa: "जलम्"          # Sanskrit
│   ├── la: "aqua"          # Latin
│   ├── egy: "𓈗"           # Egyptian hieroglyph (Gardiner N35)
│   ├── sux: "𒀀"           # Sumerian cuneiform
│   └── ...                  # Any language, any era
├── char_refs: [symlink to Character Galaxy entries for each glyph]
├── word_refs: [symlink to Word Galaxy entries]
├── taxonomy_refs: ["chemistry_compound", "physics_fluid", "biology_essential"]
└── domain: "General"  # Cross-domain concept
```

**Key:** The star IS the meaning. Languages are surface forms. The same star serves a Latin scholar, a Chinese student, and the AI — all accessing the same semantic node.

### Layer 1 Expansion: All Writing Systems

Expand Character Galaxy beyond Latin + math symbols to include ALL human writing systems:

#### Priority 1: Living Scripts (Active Unicode Blocks)
| Script | Unicode Range | Approx. Chars | Region |
|--------|-------------|---------------|--------|
| Latin Extended | U+0080-U+024F | ~300 | Global |
| Cyrillic | U+0400-U+04FF | ~256 | Russia, Eastern Europe |
| Greek | U+0370-U+03FF | ~135 | Greece, math/science |
| Arabic | U+0600-U+06FF | ~256 | Middle East, North Africa |
| Devanagari | U+0900-U+097F | ~128 | India (Hindi, Sanskrit) |
| CJK Unified | U+4E00-U+9FFF | ~20,992 | China, Japan, Korea |
| Hangul Syllables | U+AC00-U+D7AF | ~11,172 | Korea |
| Hiragana + Katakana | U+3040-U+30FF | ~192 | Japan |
| Hebrew | U+0590-U+05FF | ~88 | Israel |
| Thai | U+0E00-U+0E7F | ~87 | Thailand |
| Bengali | U+0980-U+09FF | ~96 | Bangladesh, India |
| Tamil | U+0B80-U+0BFF | ~72 | South India, Sri Lanka |

#### Priority 2: Historical Scripts
| Script | Unicode Range | Approx. Chars | Era |
|--------|-------------|---------------|-----|
| Egyptian Hieroglyphs | U+13000-U+1342F | ~1,071 | ~3200 BCE |
| Cuneiform | U+12000-U+123FF | ~1,024 | ~3400 BCE |
| Linear B | U+10000-U+1007F | ~88 | ~1450 BCE |
| Phoenician | U+10900-U+1091F | ~29 | ~1050 BCE |
| Old Persian | U+103A0-U+103DF | ~50 | ~525 BCE |
| Runic | U+16A0-U+16FF | ~89 | ~150 CE |
| Gothic | U+10330-U+1034F | ~27 | ~350 CE |
| Coptic | U+2C80-U+2CFF | ~123 | ~300 CE |
| Brahmi | U+11000-U+1107F | ~109 | ~300 BCE |

#### Priority 3: Specialized Symbol Sets
| Set | Content | Purpose |
|-----|---------|---------|
| Braille Patterns | U+2800-U+28FF (256 chars) | Accessibility |
| Musical Symbols | U+1D100-U+1D1FF | Audio Galaxy linkage |
| Alchemical Symbols | U+1F700-U+1F77F | Historical chemistry |
| Emoji | U+1F600+ | Modern communication |

**Implementation:** Each character is a Layer 1 star with:
- `form_rpn`: Procedural Bezier/segment data (how it looks)
- `script`: Unicode script name
- `codepoint`: Unicode codepoint
- `direction`: LTR / RTL / TTB
- `char_refs`: Component references (for composed characters like CJK radicals)

---

### All Numeral Systems

Expand Number Galaxy to include ALL numeral representations:

```python
# Each numeral system is a set of Grammar rules (Layer 3)
# mapping between the universal meaning (integer value) and its surface form

NUMERAL_SYSTEMS = {
    "arabic_western": {
        "name": "Western Arabic (Hindu-Arabic)",
        "base": 10,
        "positional": True,
        "chars": "0123456789",
        "surface_forms": {"en": "Western Arabic numerals"},
    },
    "arabic_eastern": {
        "name": "Eastern Arabic",
        "base": 10,
        "positional": True,
        "chars": "٠١٢٣٤٥٦٧٨٩",
        "surface_forms": {"ar": "أرقام عربية مشرقية"},
    },
    "roman": {
        "name": "Roman Numerals",
        "base": None,  # Additive/subtractive, not positional
        "positional": False,
        "symbols": {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000},
        "rules_rpn": "ROMAN_PARSE SUBTRACTIVE_RULE",  # IV=4, IX=9, etc.
    },
    "chinese_traditional": {
        "name": "Chinese Traditional",
        "base": 10,
        "positional": False,  # Multiplicative grouping
        "symbols": {"零": 0, "一": 1, "二": 2, "三": 3, "四": 4, "五": 5,
                     "六": 6, "七": 7, "八": 8, "九": 9, "十": 10,
                     "百": 100, "千": 1000, "萬": 10000},
    },
    "chinese_financial": {
        "name": "Chinese Financial (Anti-fraud)",
        "symbols": {"壹": 1, "貳": 2, "參": 3, "肆": 4, "伍": 5,
                     "陸": 6, "柒": 7, "捌": 8, "玖": 9, "拾": 10},
    },
    "devanagari": {
        "name": "Devanagari Numerals",
        "base": 10,
        "positional": True,
        "chars": "०१२३४५६७८९",
    },
    "thai": {
        "name": "Thai Numerals",
        "base": 10,
        "positional": True,
        "chars": "๐๑๒๓๔๕๖๗๘๙",
    },
    "bengali": {
        "name": "Bengali Numerals",
        "base": 10,
        "positional": True,
        "chars": "০১২৩৪৫৬৭৮৯",
    },
    "mayan": {
        "name": "Maya Vigesimal",
        "base": 20,
        "positional": True,
        "description": "Dot (1) and bar (5) system, shell for zero",
    },
    "babylonian": {
        "name": "Babylonian Sexagesimal",
        "base": 60,
        "positional": True,
        "description": "Wedge marks in base-60, cuneiform",
    },
    "egyptian": {
        "name": "Egyptian Hieroglyphic",
        "base": 10,
        "positional": False,  # Additive
        "symbols": {"𓏺": 1, "𓎆": 10, "𓍢": 100, "𓆼": 1000,
                     "𓂭": 10000, "𓆐": 100000, "𓁨": 1000000},
    },
    "greek_alphabetic": {
        "name": "Greek Alphabetic (Ionic)",
        "description": "α=1, β=2, ... ι=10, κ=20, ... ρ=100",
    },
    "hebrew_gematria": {
        "name": "Hebrew Gematria",
        "description": "א=1, ב=2, ... י=10, כ=20, ... ק=100",
    },
    "tally": {
        "name": "Tally Marks",
        "base": 5,  # Groups of 5 (four vertical + one diagonal)
        "description": "Universal counting: ||||  with diagonal strike-through at 5",
        "form_rpn": "TALLY_MARK_GROUP",
    },
    "finger_counting": {
        "name": "Finger/Hand Counting",
        "base": 10,
        "description": "Visual hand positions for 1-10, varies by culture",
    },
    "binary": {
        "name": "Binary (Base-2)",
        "base": 2,
        "positional": True,
        "chars": "01",
        "surface_forms": {"en": "Binary", "pt": "Binário"},
    },
    "hexadecimal": {
        "name": "Hexadecimal (Base-16)",
        "base": 16,
        "positional": True,
        "chars": "0123456789ABCDEF",
    },
    "octal": {
        "name": "Octal (Base-8)",
        "base": 8,
        "positional": True,
        "chars": "01234567",
    },
}
```

**Each numeral system produces:**
- Layer 1 stars: The glyph forms (visual appearance of each digit/symbol)
- Layer 2 stars: The meaning (numeric value each symbol represents)
- Layer 3 rules: Grammar for parsing/composing numbers in that system (e.g., Roman subtractive rule: IV=4)

---

## Standard Sizes, Formats, and Specifications

### Paper Sizes (ISO 216 + Regional)

All as Grammar rules / Meta-rules in the Standards domain:

```python
PAPER_SIZES = {
    # ISO 216 A-series (mm)
    "A0": (841, 1189), "A1": (594, 841), "A2": (420, 594),
    "A3": (297, 420), "A4": (210, 297), "A5": (148, 210),
    "A6": (105, 148), "A7": (74, 105), "A8": (52, 74),
    "A9": (37, 52), "A10": (26, 37),
    # ISO 216 B-series (mm)
    "B0": (1000, 1414), "B1": (707, 1000), "B2": (500, 707),
    "B3": (353, 500), "B4": (250, 353), "B5": (176, 250),
    # ISO 216 C-series (envelopes, mm)
    "C0": (917, 1297), "C3": (324, 458), "C4": (229, 324),
    "C5": (162, 229), "C6": (114, 162), "DL": (110, 220),
    # North American
    "Letter": (215.9, 279.4), "Legal": (215.9, 355.6),
    "Tabloid": (279.4, 431.8), "Ledger": (431.8, 279.4),
    "Executive": (184.2, 266.7), "Half_Letter": (139.7, 215.9),
    # Japanese JIS
    "JIS_B0": (1030, 1456), "JIS_B1": (728, 1030),
    "JIS_B2": (515, 728), "JIS_B3": (364, 515),
    "JIS_B4": (257, 364), "JIS_B5": (182, 257),
}

# Relationship rule: A-series follows √2 ratio
# A(n+1) width = A(n) height / 2, A(n+1) height = A(n) width
# Area(A0) = 1 m²
```

### Book Sizes (Standard Trim Sizes)

```python
BOOK_SIZES = {
    "mass_market_paperback": (108, 175),   # mm, 4.25" × 6.875"
    "trade_paperback_small": (127, 203),   # 5" × 8"
    "trade_paperback": (140, 216),         # 5.5" × 8.5"
    "trade_paperback_large": (152, 229),   # 6" × 9"
    "hardcover_small": (140, 216),         # 5.5" × 8.5"
    "hardcover_standard": (152, 229),      # 6" × 9"
    "hardcover_large": (178, 254),         # 7" × 10"
    "textbook": (203, 254),               # 8" × 10"
    "coffee_table": (254, 305),           # 10" × 12"
    "quarto": (190, 250),                 # Traditional quarto
    "octavo": (152, 229),                 # Traditional octavo
    "folio": (305, 483),                  # Traditional folio
    "pamphlet": (140, 216),               # Standard pamphlet
    "leaflet": (99, 210),                 # DL-size leaflet (folded A4)
    "broadsheet": (375, 600),             # Newspaper broadsheet
    "tabloid_newspaper": (280, 430),      # Newspaper tabloid
}
```

### Standard Open Formats (File Extensions)

```python
STANDARD_FORMATS = {
    # 3D
    "glb": {"mime": "model/gltf-binary", "domain": "Visual", "description": "GL Transmission Format Binary"},
    "gltf": {"mime": "model/gltf+json", "domain": "Visual", "description": "GL Transmission Format JSON"},
    "obj": {"mime": "model/obj", "domain": "Visual", "description": "Wavefront OBJ"},
    "fbx": {"mime": "application/octet-stream", "domain": "Visual", "description": "Filmbox"},
    "usdz": {"mime": "model/vnd.usdz+zip", "domain": "Visual", "description": "Universal Scene Description"},
    "stl": {"mime": "model/stl", "domain": "Visual", "description": "Stereolithography"},
    # Document
    "pdf": {"mime": "application/pdf", "domain": "Language", "description": "Portable Document Format"},
    "html": {"mime": "text/html", "domain": "Language", "description": "HyperText Markup Language"},
    "md": {"mime": "text/markdown", "domain": "Language", "description": "Markdown"},
    "tex": {"mime": "application/x-tex", "domain": "Mathematics", "description": "LaTeX"},
    "epub": {"mime": "application/epub+zip", "domain": "Language", "description": "Electronic Publication"},
    # Image
    "png": {"mime": "image/png", "domain": "Visual", "description": "Portable Network Graphics"},
    "svg": {"mime": "image/svg+xml", "domain": "Visual", "description": "Scalable Vector Graphics"},
    "jpg": {"mime": "image/jpeg", "domain": "Visual", "description": "JPEG Image"},
    "webp": {"mime": "image/webp", "domain": "Visual", "description": "WebP Image"},
    # Audio
    "wav": {"mime": "audio/wav", "domain": "Audio", "description": "Waveform Audio"},
    "mp3": {"mime": "audio/mpeg", "domain": "Audio", "description": "MPEG Audio Layer 3"},
    "ogg": {"mime": "audio/ogg", "domain": "Audio", "description": "Ogg Vorbis"},
    "flac": {"mime": "audio/flac", "domain": "Audio", "description": "Free Lossless Audio Codec"},
    "opus": {"mime": "audio/opus", "domain": "Audio", "description": "Opus Audio"},
    # Video
    "mp4": {"mime": "video/mp4", "domain": "Audio", "description": "MPEG-4 Video"},
    "webm": {"mime": "video/webm", "domain": "Audio", "description": "WebM Video"},
    "mkv": {"mime": "video/x-matroska", "domain": "Audio", "description": "Matroska Video"},
    # Data
    "json": {"mime": "application/json", "domain": "Tools", "description": "JavaScript Object Notation"},
    "csv": {"mime": "text/csv", "domain": "Tools", "description": "Comma-Separated Values"},
    "xml": {"mime": "application/xml", "domain": "Tools", "description": "Extensible Markup Language"},
    "yaml": {"mime": "application/x-yaml", "domain": "Tools", "description": "YAML Ain't Markup Language"},
    # Code
    "py": {"mime": "text/x-python", "domain": "Tools", "description": "Python Source"},
    "ts": {"mime": "text/typescript", "domain": "Tools", "description": "TypeScript Source"},
    "rs": {"mime": "text/x-rust", "domain": "Tools", "description": "Rust Source"},
    "cu": {"mime": "text/x-cuda", "domain": "Tools", "description": "CUDA Source"},
    "ptx": {"mime": "text/x-ptx", "domain": "Tools", "description": "Parallel Thread Execution"},
}
```

---

## Measurements and Units (Wikipedia-Sourced)

### Structure

All measurement knowledge follows the 4-layer pattern:

- **Layer 1 (Form):** The symbol/glyph (°C, kg, Pa, etc.)
- **Layer 2 (Meaning):** What it measures (temperature, mass, pressure)
- **Layer 3 (Rules):** Conversion formulas as RPN programs
- **Layer 4 (Meta-Rules):** When to use which unit system (SI vs Imperial vs historical)

### Unit Domains

```python
MEASUREMENT_DOMAINS = {
    "length": {
        "si_base": "metre",
        "units": {
            "metre": {"symbol": "m", "to_si": "1.0 MUL"},
            "kilometre": {"symbol": "km", "to_si": "1000.0 MUL"},
            "centimetre": {"symbol": "cm", "to_si": "0.01 MUL"},
            "millimetre": {"symbol": "mm", "to_si": "0.001 MUL"},
            "micrometre": {"symbol": "μm", "to_si": "1e-6 MUL"},
            "nanometre": {"symbol": "nm", "to_si": "1e-9 MUL"},
            "picometre": {"symbol": "pm", "to_si": "1e-12 MUL"},
            "inch": {"symbol": "in", "to_si": "0.0254 MUL"},
            "foot": {"symbol": "ft", "to_si": "0.3048 MUL"},
            "yard": {"symbol": "yd", "to_si": "0.9144 MUL"},
            "mile": {"symbol": "mi", "to_si": "1609.344 MUL"},
            "nautical_mile": {"symbol": "nmi", "to_si": "1852.0 MUL"},
            "astronomical_unit": {"symbol": "AU", "to_si": "1.496e11 MUL"},
            "light_year": {"symbol": "ly", "to_si": "9.461e15 MUL"},
            "parsec": {"symbol": "pc", "to_si": "3.086e16 MUL"},
            "angstrom": {"symbol": "Å", "to_si": "1e-10 MUL"},
            # Historical
            "cubit": {"symbol": "cubit", "to_si": "0.4572 MUL", "era": "ancient"},
            "fathom": {"symbol": "ftm", "to_si": "1.8288 MUL"},
            "furlong": {"symbol": "fur", "to_si": "201.168 MUL"},
            "league": {"symbol": "lea", "to_si": "4828.032 MUL"},
        },
    },
    "mass": {
        "si_base": "kilogram",
        "units": {
            "kilogram": {"symbol": "kg", "to_si": "1.0 MUL"},
            "gram": {"symbol": "g", "to_si": "0.001 MUL"},
            "milligram": {"symbol": "mg", "to_si": "1e-6 MUL"},
            "tonne": {"symbol": "t", "to_si": "1000.0 MUL"},
            "pound": {"symbol": "lb", "to_si": "0.453592 MUL"},
            "ounce": {"symbol": "oz", "to_si": "0.0283495 MUL"},
            "stone": {"symbol": "st", "to_si": "6.35029 MUL"},
            "atomic_mass_unit": {"symbol": "u", "to_si": "1.66054e-27 MUL"},
            "solar_mass": {"symbol": "M☉", "to_si": "1.989e30 MUL"},
            "carat": {"symbol": "ct", "to_si": "0.0002 MUL"},
            "grain": {"symbol": "gr", "to_si": "6.47989e-5 MUL"},
        },
    },
    "temperature": {
        "si_base": "kelvin",
        "units": {
            "kelvin": {"symbol": "K", "to_si": "1.0 MUL"},
            "celsius": {"symbol": "°C", "to_si": "273.15 ADD"},
            "fahrenheit": {"symbol": "°F", "to_si": "32 SUB 5 MUL 9 DIV 273.15 ADD"},
            "rankine": {"symbol": "°R", "to_si": "5 MUL 9 DIV"},
        },
    },
    "pressure": {
        "si_base": "pascal",
        "units": {
            "pascal": {"symbol": "Pa", "to_si": "1.0 MUL"},
            "kilopascal": {"symbol": "kPa", "to_si": "1000.0 MUL"},
            "megapascal": {"symbol": "MPa", "to_si": "1e6 MUL"},
            "bar": {"symbol": "bar", "to_si": "1e5 MUL"},
            "atmosphere": {"symbol": "atm", "to_si": "101325.0 MUL"},
            "torr": {"symbol": "Torr", "to_si": "133.322 MUL"},
            "psi": {"symbol": "psi", "to_si": "6894.76 MUL"},
            "mmHg": {"symbol": "mmHg", "to_si": "133.322 MUL"},
        },
    },
    "time": {
        "si_base": "second",
        "units": {
            "second": {"symbol": "s", "to_si": "1.0 MUL"},
            "millisecond": {"symbol": "ms", "to_si": "0.001 MUL"},
            "microsecond": {"symbol": "μs", "to_si": "1e-6 MUL"},
            "nanosecond": {"symbol": "ns", "to_si": "1e-9 MUL"},
            "minute": {"symbol": "min", "to_si": "60.0 MUL"},
            "hour": {"symbol": "h", "to_si": "3600.0 MUL"},
            "day": {"symbol": "d", "to_si": "86400.0 MUL"},
            "week": {"symbol": "wk", "to_si": "604800.0 MUL"},
            "year_julian": {"symbol": "a", "to_si": "31557600.0 MUL"},
            "planck_time": {"symbol": "tₚ", "to_si": "5.391e-44 MUL"},
        },
    },
    "electric_current": {"si_base": "ampere", "units": {"ampere": {"symbol": "A"}, "milliampere": {"symbol": "mA"}}},
    "voltage": {"si_base": "volt", "units": {"volt": {"symbol": "V"}, "millivolt": {"symbol": "mV"}, "kilovolt": {"symbol": "kV"}}},
    "energy": {
        "si_base": "joule",
        "units": {
            "joule": {"symbol": "J", "to_si": "1.0 MUL"},
            "kilojoule": {"symbol": "kJ", "to_si": "1000.0 MUL"},
            "calorie": {"symbol": "cal", "to_si": "4.184 MUL"},
            "kilocalorie": {"symbol": "kcal", "to_si": "4184.0 MUL"},
            "electronvolt": {"symbol": "eV", "to_si": "1.602e-19 MUL"},
            "kilowatt_hour": {"symbol": "kWh", "to_si": "3.6e6 MUL"},
            "btu": {"symbol": "BTU", "to_si": "1055.06 MUL"},
            "erg": {"symbol": "erg", "to_si": "1e-7 MUL"},
        },
    },
    "force": {
        "si_base": "newton",
        "units": {
            "newton": {"symbol": "N", "to_si": "1.0 MUL"},
            "kilonewton": {"symbol": "kN", "to_si": "1000.0 MUL"},
            "dyne": {"symbol": "dyn", "to_si": "1e-5 MUL"},
            "pound_force": {"symbol": "lbf", "to_si": "4.44822 MUL"},
        },
    },
    "frequency": {"si_base": "hertz", "units": {"hertz": {"symbol": "Hz"}, "kilohertz": {"symbol": "kHz"}, "megahertz": {"symbol": "MHz"}, "gigahertz": {"symbol": "GHz"}}},
    "area": {"si_base": "square_metre", "units": {"square_metre": {"symbol": "m²"}, "hectare": {"symbol": "ha", "to_si": "10000.0 MUL"}, "acre": {"symbol": "ac", "to_si": "4046.86 MUL"}}},
    "volume": {"si_base": "cubic_metre", "units": {"cubic_metre": {"symbol": "m³"}, "litre": {"symbol": "L", "to_si": "0.001 MUL"}, "gallon_us": {"symbol": "gal", "to_si": "0.003785 MUL"}, "gallon_uk": {"symbol": "gal(UK)", "to_si": "0.004546 MUL"}}},
    "speed": {"si_base": "metre_per_second", "units": {"metre_per_second": {"symbol": "m/s"}, "kilometre_per_hour": {"symbol": "km/h", "to_si": "0.27778 MUL"}, "mile_per_hour": {"symbol": "mph", "to_si": "0.44704 MUL"}, "knot": {"symbol": "kn", "to_si": "0.51444 MUL"}, "speed_of_light": {"symbol": "c", "to_si": "2.998e8 MUL"}}},
    "angle": {"si_base": "radian", "units": {"radian": {"symbol": "rad"}, "degree": {"symbol": "°", "to_si": "0.017453 MUL"}, "arcminute": {"symbol": "′"}, "arcsecond": {"symbol": "″"}, "gradian": {"symbol": "gon"}}},
    "luminous_intensity": {"si_base": "candela", "units": {"candela": {"symbol": "cd"}, "lumen": {"symbol": "lm"}, "lux": {"symbol": "lx"}}},
    "amount_of_substance": {"si_base": "mole", "units": {"mole": {"symbol": "mol"}}},
    "data_storage": {"si_base": "byte", "units": {"bit": {"symbol": "b"}, "byte": {"symbol": "B"}, "kilobyte": {"symbol": "KB"}, "megabyte": {"symbol": "MB"}, "gigabyte": {"symbol": "GB"}, "terabyte": {"symbol": "TB"}, "petabyte": {"symbol": "PB"}}},
}
```

**Each unit produces:**
- Layer 1 star: Symbol glyph (kg, °C, Pa)
- Layer 2 star: Meaning (what this unit measures, its physical quantity)
- Layer 3 rule: Conversion RPN program (to/from SI base)
- Layer 4 meta-rule: When to use (SI context, Imperial context, historical context)

### Physical Constants (Wikipedia-Sourced)

```python
PHYSICAL_CONSTANTS = {
    "speed_of_light": {"symbol": "c", "value": 2.99792458e8, "unit": "m/s", "exact": True},
    "gravitational_constant": {"symbol": "G", "value": 6.67430e-11, "unit": "m³/(kg·s²)"},
    "planck_constant": {"symbol": "h", "value": 6.62607015e-34, "unit": "J·s", "exact": True},
    "boltzmann_constant": {"symbol": "k_B", "value": 1.380649e-23, "unit": "J/K", "exact": True},
    "avogadro_number": {"symbol": "N_A", "value": 6.02214076e23, "unit": "1/mol", "exact": True},
    "elementary_charge": {"symbol": "e", "value": 1.602176634e-19, "unit": "C", "exact": True},
    "vacuum_permittivity": {"symbol": "ε₀", "value": 8.8541878128e-12, "unit": "F/m"},
    "vacuum_permeability": {"symbol": "μ₀", "value": 1.25663706212e-6, "unit": "N/A²"},
    "electron_mass": {"symbol": "m_e", "value": 9.1093837015e-31, "unit": "kg"},
    "proton_mass": {"symbol": "m_p", "value": 1.67262192369e-27, "unit": "kg"},
    "neutron_mass": {"symbol": "m_n", "value": 1.67492749804e-27, "unit": "kg"},
    "fine_structure_constant": {"symbol": "α", "value": 7.2973525693e-3, "unit": "dimensionless"},
    "gas_constant": {"symbol": "R", "value": 8.314462618, "unit": "J/(mol·K)", "exact": True},
    "stefan_boltzmann": {"symbol": "σ", "value": 5.670374419e-8, "unit": "W/(m²·K⁴)"},
    "standard_gravity": {"symbol": "g", "value": 9.80665, "unit": "m/s²", "exact": True},
    "standard_atmosphere": {"symbol": "atm", "value": 101325.0, "unit": "Pa", "exact": True},
}
```

---

## Periodic Table and Materials Science

### Periodic Table (All 118 Elements)

Each element is a **meaning-centric star** in the Reality Galaxy:

```python
# Schema for each element star
@dataclass
class ElementEntry:
    atomic_number: int          # 1-118
    symbol: str                 # "H", "He", "Li", ...
    name_en: str                # English name
    surface_forms: dict         # {"en": "Hydrogen", "pt": "Hidrogénio", "la": "Hydrogenium", ...}
    atomic_mass: float          # In atomic mass units (u)
    category: str               # "nonmetal", "noble_gas", "alkali_metal", "transition_metal", ...
    group: int | None           # 1-18 (or None for lanthanides/actinides)
    period: int                 # 1-7
    block: str                  # "s", "p", "d", "f"
    electron_config: str        # "1s¹", "1s² 2s²", ...
    electronegativity: float | None  # Pauling scale
    density: float | None       # kg/m³ at STP
    melting_point: float | None # K
    boiling_point: float | None # K
    state_at_stp: str           # "solid", "liquid", "gas"
    discovery_year: int | None
    # Layer 3 rules
    oxidation_states: list[int] # Common oxidation states
    bonds_rpn: str              # RPN program for valence/bonding rules
    # Relationships
    taxonomy_refs: list[str]    # ["chemistry_element", "physics_atom", ...]
```

### Atomic Structure (Subatomic Particles)

```python
SUBATOMIC_PARTICLES = {
    "proton": {"charge": +1, "mass_u": 1.007276, "quarks": "uud", "baryon": True},
    "neutron": {"charge": 0, "mass_u": 1.008665, "quarks": "udd", "baryon": True},
    "electron": {"charge": -1, "mass_u": 0.000549, "lepton": True},
    "photon": {"charge": 0, "mass_u": 0, "boson": True, "force": "electromagnetic"},
    "gluon": {"charge": 0, "mass_u": 0, "boson": True, "force": "strong"},
    "w_plus": {"charge": +1, "mass_u": 86.1, "boson": True, "force": "weak"},
    "w_minus": {"charge": -1, "mass_u": 86.1, "boson": True, "force": "weak"},
    "z_boson": {"charge": 0, "mass_u": 97.3, "boson": True, "force": "weak"},
    "higgs": {"charge": 0, "mass_u": 134.0, "boson": True, "force": "mass"},
    # Quarks
    "up": {"charge": 2/3, "generation": 1}, "down": {"charge": -1/3, "generation": 1},
    "charm": {"charge": 2/3, "generation": 2}, "strange": {"charge": -1/3, "generation": 2},
    "top": {"charge": 2/3, "generation": 3}, "bottom": {"charge": -1/3, "generation": 3},
}
```

### Materials Science (How Materials Are Made)

Layer 3 rules for material composition:

```python
# Atoms → Elements → Compounds → Materials → Objects
# Each level is a Grammar rule that composes lower-level stars

MATERIAL_RULES = {
    "water_composition": {
        "formula": "H₂O",
        "rule_rpn": "ELEMENT_H 2 BOND_COVALENT ELEMENT_O 1 BOND_COVALENT MOLECULE_COMPOSE",
        "elements": ["H", "O"],
        "bond_angle": 104.5,  # degrees
        "state_rules": {"below_273K": "solid", "273K_to_373K": "liquid", "above_373K": "gas"},
    },
    "steel_composition": {
        "formula": "Fe + C (0.2-2.1%)",
        "rule_rpn": "ELEMENT_Fe ELEMENT_C 0.02 ALLOY_MIX",
        "category": "alloy",
        "properties": {"hardness": "high", "ductility": "medium", "conductivity": "high"},
    },
    "glass_composition": {
        "formula": "SiO₂ (primary) + Na₂O + CaO",
        "rule_rpn": "ELEMENT_Si ELEMENT_O 2 BOND_COVALENT ELEMENT_Na 2 ELEMENT_O FLUX_ADD",
        "category": "amorphous_solid",
    },
    "wood_composition": {
        "components": ["cellulose (40-50%)", "hemicellulose (20-30%)", "lignin (25-35%)"],
        "category": "natural_composite",
        "source": "biology_plant",
    },
    "concrete_composition": {
        "components": ["cement (Portland)", "water", "aggregate (sand + gravel)"],
        "rule_rpn": "CEMENT WATER 0.45 RATIO_MIX AGGREGATE ADD HYDRATION_CURE",
        "category": "composite",
    },
}
```

---

## Proceduralization: The Augmentation-to-Symlink Pipeline

Daniel's key insight: **"Proceduralize means to transform content into symlink content. Word by word, you transform it into the symlink of the meaning."**

This is the bridge between H16 augmentation and Galaxy population:

```
Raw Content (text, audio transcript, etc.)
    ↓ [H16 augmentation providers]
Augmented JSON (summary, entities, relationships, domain)
    ↓ [H17 proceduralization]
Symlinked Star Content:
    - Each word → symlink to Word Galaxy meaning star
    - Each entity → symlink to Entity Galaxy star
    - Each relationship → Grammar rule connecting stars
    - Each measurement → symlink to Unit star + value
    - Each element/compound → symlink to Periodic Table star
    ↓ [content_to_stars.py]
House Objects (books, shelves, displays)
```

### Proceduralization Function

```python
def proceduralize_content(
    augmented: AugmentationResult,
    *,
    galaxy_lookup: Callable[[str], str | None],  # word → star_id or None
) -> list[dict]:
    """Transform augmented content into symlink-based star references.

    For each word in the summary/entities:
    1. Look up the word in Word Galaxy → get meaning star_id
    2. If found, replace word with symlink reference
    3. If not found, create a new candidate star (for sleep-time review)

    Returns list of symlink entries ready for content_to_stars.
    """
```

---

## Implementation Plan

### Submodule Structure

```
knowledge3d/ingestion/universal_knowledge/
├── __init__.py
├── writing_systems.py      # All Unicode scripts → Character Galaxy entries
├── numeral_systems.py      # All numeral systems → Number Galaxy entries + Grammar rules
├── paper_and_book_sizes.py # ISO 216, trim sizes, format specs → Standards stars
├── file_formats.py         # Open format registry → Tools Galaxy entries
├── measurements.py         # All units + conversion RPN → Reality Galaxy
├── physical_constants.py   # Fundamental constants → Reality Galaxy
├── periodic_table.py       # 118 elements + subatomic → Reality Galaxy
├── materials_science.py    # Compounds, alloys, materials → Reality Galaxy rules
├── proceduralize.py        # Word-by-word symlink transformation
└── wikipedia_ingest.py     # Wikipedia dataset connector for bulk ingestion
```

### Data Sources

| Domain | Source | License |
|--------|--------|---------|
| Writing systems | Unicode CLDR + character database | Unicode ToS |
| Numeral systems | Unicode + Wikipedia | CC-BY-SA |
| Paper/book sizes | ISO 216 (public data), Wikipedia | CC-BY-SA |
| File formats | IANA media types registry | Public |
| Measurements | BIPM SI Brochure + Wikipedia | CC-BY-SA / Public |
| Physical constants | CODATA 2018 (NIST) | Public domain |
| Periodic table | IUPAC + Wikipedia | CC-BY-SA / Public |
| Materials | Wikipedia + CRC Handbook (public data) | CC-BY-SA |

---

## Tests

```python
# test_universal_knowledge.py

def test_all_writing_systems_have_unicode_range():
    """Every script entry has valid Unicode start/end codepoints."""

def test_numeral_systems_round_trip():
    """For positional systems: encode(42) → decode → 42."""

def test_paper_sizes_a_series_ratio():
    """A(n+1) dimensions follow √2 ratio within 1mm tolerance."""

def test_unit_conversion_round_trip():
    """Convert 100°F → K → °C → °F should return ~100."""

def test_periodic_table_complete():
    """All 118 elements present with atomic number, symbol, mass."""

def test_periodic_table_groups():
    """Elements assigned to correct groups/periods/blocks."""

def test_material_composition_references_elements():
    """Every material rule references valid element symbols."""

def test_proceduralize_creates_symlinks():
    """Word-by-word transformation produces star_id references."""

def test_physical_constants_exact_values():
    """2019 SI redefinition constants match exact defined values."""
```

---

## Success Criteria

1. Character Galaxy expanded to cover all Priority 1 living scripts
2. All numeral systems produce valid Layer 1/2/3 entries
3. Paper sizes, book sizes, file formats registered as standards
4. All SI base units + common derived units with RPN conversion programs
5. Complete periodic table (118 elements) with properties and surface forms
6. Materials science rules compose elements into compounds
7. Physical constants match CODATA/NIST values
8. Proceduralization pipeline transforms words to meaning symlinks
9. All data sources properly attributed with licenses
10. Tests validate completeness and round-trip conversions
