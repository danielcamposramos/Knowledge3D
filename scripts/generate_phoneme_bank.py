#!/usr/bin/env python3
"""
Generate a small open phoneme audio bank using espeak-ng for the supported languages.

Outputs per-phoneme WAVs under:
    /K3D/K3D_llama_cpp/datasets/audio/phoneme_bank/<lang>/<phoneme>.wav

Languages covered (basic sets):
    - en (English)
    - pt (Portuguese)
    - es (Spanish)
    - zh (Mandarin, using pinyin syllable set)

Requires: espeak or espeak-ng in PATH.

Usage:
    PYTHONPATH=. python3 scripts/generate_phoneme_bank.py --langs en pt es zh
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
from typing import Dict, List

OUTPUT_ROOT = Path("/K3D/K3D_llama_cpp/datasets/audio/phoneme_bank")

# Simplified phoneme/pinyin inventories (can be expanded later).
PHONEMES: Dict[str, List[str]] = {
    "en": [
        "i", "ɪ", "eɪ", "ɛ", "æ", "ɑ", "ɔ", "oʊ", "ʊ", "u",
        "ʌ", "ə", "ɝ",
        "p", "b", "t", "d", "k", "g", "f", "v", "θ", "ð", "s", "z",
        "ʃ", "ʒ", "h", "m", "n", "ŋ", "l", "r", "w", "j", "tʃ", "dʒ",
    ],
    "pt": [
        "a", "ɑ̃", "e", "ɛ", "i", "o", "ɔ", "u", "ɐ", "ɐ̃", "õ", "ũ",
        "p", "b", "t", "d", "k", "g", "f", "v", "s", "z", "ʃ", "ʒ", "h",
        "m", "n", "ɲ", "l", "ʎ", "ɾ", "x", "ɣ",
    ],
    "es": [
        "a", "e", "i", "o", "u",
        "p", "b", "t", "d", "k", "g", "f", "β", "θ", "ð", "s", "x",
        "m", "n", "ɲ", "l", "ʎ", "r", "ɾ", "tʃ", "ʝ",
    ],
    "zh": [
        # Basic pinyin syllables (initial+final combos); kept small.
        "ma", "ba", "pa", "da", "ta", "na", "la", "ga", "ka",
        "zhi", "chi", "shi", "ri", "zi", "ci", "si",
        "zhong", "cheng", "shen", "ren", "zong", "cong", "song",
        "pang", "mang", "fang", "wang", "yang",
        "yi", "ya", "yao", "you", "ye", "yan", "yang", "yong", "yin", "ying",
    ],
}


def synthesize(lang: str, token: str, out_path: Path, speed: int = 140, rate: int = 16000) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "espeak",
        "-v",
        lang,
        "-s",
        str(speed),
        "-w",
        str(out_path),
        token,
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--langs", nargs="+", default=["en", "pt", "es", "zh"], help="Languages to generate")
    ap.add_argument("--speed", type=int, default=140, help="espeak speed")
    ap.add_argument("--rate", type=int, default=16000, help="sample rate (via espeak)")
    args = ap.parse_args()

    for lang in args.langs:
        if lang not in PHONEMES:
            print(f"[warn] lang {lang} not supported, skipping")
            continue
        for ph in PHONEMES[lang]:
            out_path = OUTPUT_ROOT / lang / f"{ph.replace(' ', '_')}.wav"
            try:
                synthesize(lang, ph, out_path, speed=args.speed, rate=args.rate)
            except Exception as exc:
                print(f"[warn] failed to synthesize {lang}:{ph}: {exc}")
    print(f"Phoneme bank generated under {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
