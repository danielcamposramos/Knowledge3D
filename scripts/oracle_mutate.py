#!/usr/bin/env python3
"""
Phase 5.0 Oracle Mutator: deterministic template mutations for new problems.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


NUMBER_PATTERN = re.compile(r"(?P<num>-?\d+(?:\.\d+)?(?:/\d+(?:\.\d+)?)?)")


ENTITY_PAIRS = [
    ("apple", "orange"),
    ("banana", "pear"),
    ("bob", "alice"),
    ("john", "mary"),
    ("cat", "dog"),
    ("left", "right"),
    ("north", "south"),
    ("east", "west"),
    ("increase", "decrease"),
    ("add", "subtract"),
]

CALCULUS_EXPR_PATTERNS = [
    re.compile(r"f\(x\)\s*=\s*(?P<expr>[^,.;]+)", re.IGNORECASE),
    re.compile(
        r"(?:find|evaluate)?\s*derivative of\s+(?P<expr>.+?)(?:\s+at\s+|[?.!]|$)",
        re.IGNORECASE,
    ),
]


def _iter_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def _stable_template_id(text: str, source: str, label: Optional[int]) -> str:
    digest = hashlib.sha256(f"{text}|{source}|{label}".encode("utf-8")).hexdigest()
    return digest[:16]


def _stable_candidate_id(template_id: str, mutation_type: str, generated_text: str) -> str:
    digest = hashlib.sha256(f"{template_id}|{mutation_type}|{generated_text}".encode("utf-8")).hexdigest()
    return digest[:16]


def _format_number(value: float, *, had_decimal: bool, was_fraction: bool) -> str:
    if was_fraction or had_decimal:
        formatted = f"{value:.3f}".rstrip("0").rstrip(".")
        if formatted == "-0":
            return "0"
        return formatted
    if abs(value - round(value)) < 1e-6:
        return str(int(round(value)))
    formatted = f"{value:.3f}".rstrip("0").rstrip(".")
    if formatted == "-0":
        return "0"
    return formatted


def _scale_number(token: str, scale: float) -> Tuple[str, float]:
    was_fraction = "/" in token
    had_decimal = "." in token
    if was_fraction:
        try:
            numerator, denominator = token.split("/", 1)
            value = float(numerator) / float(denominator)
        except (ValueError, ZeroDivisionError):
            value = float(token)
    else:
        value = float(token)
    scaled = value * scale
    return _format_number(scaled, had_decimal=had_decimal, was_fraction=was_fraction), scaled


def _mutate_numeric_scaling(text: str, rng: random.Random) -> Tuple[str, Dict[str, Any]]:
    scales: List[float] = []
    scaled_values: List[float] = []

    def _replace(match: re.Match) -> str:
        token = match.group("num")
        scale = rng.uniform(0.5, 2.0)
        replacement, scaled = _scale_number(token, scale)
        scales.append(scale)
        scaled_values.append(scaled)
        return replacement

    mutated = NUMBER_PATTERN.sub(_replace, text)
    metadata = {
        "numeric_scaling": {
            "count": len(scales),
            "scales": scales,
            "scaled_values": scaled_values,
        }
    }
    return mutated, metadata


def _apply_entity_swap(text: str, rng: random.Random) -> Tuple[str, Optional[Dict[str, str]]]:
    candidates = list(ENTITY_PAIRS)
    rng.shuffle(candidates)
    for left, right in candidates:
        for src, dst in ((left, right), (right, left)):
            pattern = re.compile(rf"\\b{re.escape(src)}\\b", re.IGNORECASE)
            if not pattern.search(text):
                continue
            def _repl(match: re.Match) -> str:
                original = match.group(0)
                if original.isupper():
                    return dst.upper()
                if original[:1].isupper():
                    return dst[:1].upper() + dst[1:]
                return dst

            return pattern.sub(_repl, text), {"from": src, "to": dst}
    return text, None


def _complexify_calculus(text: str, rng: random.Random) -> Tuple[str, Optional[Dict[str, Any]]]:
    match = None
    for pattern in CALCULUS_EXPR_PATTERNS:
        match = pattern.search(text)
        if match:
            break
    if not match:
        return text, None

    expr = match.group("expr").strip()
    if not expr:
        return text, None

    choice = rng.random()
    meta: Dict[str, Any] = {}

    if choice < 0.34:
        coeff = rng.uniform(1.0, 6.0)
        power = rng.randint(2, 4)
        sign = "-" if rng.random() < 0.4 else "+"
        term = f"{coeff:.3f}".rstrip("0").rstrip(".") + f"x^{power}"
        new_expr = f"({expr}) {sign} {term}"
        meta["type"] = "add_term"
        meta["term"] = f"{sign} {term}"
    elif choice < 0.67:
        shift = rng.uniform(0.5, 4.0)
        shift_str = f"{shift:.3f}".rstrip("0").rstrip(".")
        new_expr = f"(x+{shift_str})*({expr})"
        meta["type"] = "wrap_product"
        meta["shift"] = shift_str
    else:
        func = rng.choice(["sin", "cos", "exp"])
        new_expr = f"{func}({expr})"
        meta["type"] = "nested_function"
        meta["function"] = func

    start, end = match.span("expr")
    mutated = text[:start] + new_expr + text[end:]
    return mutated, meta


def _complexify_gsm8k(text: str, rng: random.Random) -> Tuple[str, Optional[Dict[str, Any]]]:
    numbers = [m.group("num") for m in NUMBER_PATTERN.finditer(text)]
    base = None
    add = None
    sub = None
    if len(numbers) >= 1:
        try:
            base = int(float(numbers[0]))
        except ValueError:
            base = None
    if len(numbers) >= 2:
        try:
            add = int(float(numbers[1]))
        except ValueError:
            add = None
    if len(numbers) >= 3:
        try:
            sub = int(float(numbers[2]))
        except ValueError:
            sub = None

    if base is None:
        base = rng.randint(5, 25)
    if add is None:
        add = rng.randint(2, 12)
    if sub is None:
        sub = rng.randint(1, 10)
    add2 = rng.randint(2, 12)
    sub2 = rng.randint(1, 10)

    subject = rng.choice(["Mia", "Leo", "Ava", "Noah", "Zoe"])
    item = rng.choice(["coins", "marbles", "apples", "stickers"])
    template = rng.choice(["add_then_sub_twice", "sub_then_add_twice"])
    if template == "add_then_sub_twice":
        mutated = (
            f"{subject} has {base} {item}. "
            f"{subject} gets {add} more, then gives away {sub}. "
            f"{subject} gets {add2} more, then gives away {sub2}. "
            f"How many {item} are remaining altogether?"
        )
        meta = {
            "type": "micro_template",
            "steps": ["add", "sub", "add", "sub"],
            "base": base,
            "add": add,
            "sub": sub,
            "add2": add2,
            "sub2": sub2,
        }
    else:
        mutated = (
            f"There are {base} {item}. "
            f"{sub} {item} are given away, then {add} more are added. "
            f"Later {sub2} are given away, then {add2} more are added. "
            f"How many {item} are there altogether remaining?"
        )
        meta = {
            "type": "micro_template",
            "steps": ["sub", "add", "sub", "add"],
            "base": base,
            "add": add,
            "sub": sub,
            "add2": add2,
            "sub2": sub2,
        }

    return mutated, meta


def _complexify_problem(text: str, source: str, rng: random.Random) -> Tuple[str, Optional[Dict[str, Any]]]:
    if source == "calculus":
        return _complexify_calculus(text, rng)
    if source == "gsm8k":
        return _complexify_gsm8k(text, rng)
    return text, None


def _entry_rng(base_seed: int, template_id: str, mutation_index: int) -> random.Random:
    digest = hashlib.sha256(f"{base_seed}|{template_id}|{mutation_index}".encode("utf-8")).digest()
    seed = int.from_bytes(digest[:8], "little", signed=False)
    return random.Random(seed)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Oracle candidates via deterministic mutation.")
    parser.add_argument("--input", default="data/router_train.jsonl", help="Seed JSONL path.")
    parser.add_argument("--output", default="data/oracle_candidates_v1.jsonl", help="Output JSONL path.")
    parser.add_argument("--seed", type=int, default=2026, help="Base RNG seed.")
    parser.add_argument("--per-seed", type=int, default=1, help="Mutations per seed entry.")
    parser.add_argument("--max-entries", type=int, default=0, help="Limit number of seed entries (0 = all).")
    parser.add_argument("--enable-entity-swap", action="store_true", help="Enable optional entity swaps.")
    parser.add_argument("--entity-swap-prob", type=float, default=0.25, help="Probability of entity swap per mutation.")
    parser.add_argument(
        "--complexify-prob",
        type=float,
        default=0.35,
        help="Probability of structural complexifier mutation.",
    )
    args = parser.parse_args()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    total_written = 0
    seeds_seen = 0

    with output.open("w", encoding="utf-8") as handle:
        for entry in _iter_jsonl(args.input):
            if args.max_entries and seeds_seen >= args.max_entries:
                break
            text = str(entry.get("text") or "").strip()
            if not text:
                continue
            source = str(entry.get("source") or "unknown")
            label = entry.get("label")
            template_id = _stable_template_id(text, source, label)

            for mutation_index in range(int(args.per_seed)):
                rng = _entry_rng(args.seed, template_id, mutation_index)
                mutated_text, meta = _mutate_numeric_scaling(text, rng)
                mutation_type = "numeric_scaling"

                complex_meta = None
                if rng.random() < float(args.complexify_prob):
                    mutated_text, complex_meta = _complexify_problem(mutated_text, source, rng)
                    if complex_meta:
                        mutation_type = "complexifier"
                        meta["complexifier"] = complex_meta

                entity_swap_meta = None
                if args.enable_entity_swap and rng.random() < float(args.entity_swap_prob):
                    mutated_text, entity_swap_meta = _apply_entity_swap(mutated_text, rng)
                    if entity_swap_meta:
                        if mutation_type == "complexifier":
                            mutation_type = "complexifier+entity_swap"
                        else:
                            mutation_type = "numeric_scaling+entity_swap"
                        meta["entity_swap"] = entity_swap_meta

                candidate_id = _stable_candidate_id(template_id, mutation_type, mutated_text)
                payload = {
                    "candidate_id": candidate_id,
                    "template_id": template_id,
                    "mutation_type": mutation_type,
                    "generated_text": mutated_text,
                    "source_text": text,
                    "source": source,
                    "label": label,
                    "mutation_index": mutation_index,
                    "seed": args.seed,
                    "mutation_params": meta,
                }
                handle.write(json.dumps(payload, ensure_ascii=True) + "\n")
                total_written += 1

            seeds_seen += 1

    print(f"[OracleMutate] Seeds: {seeds_seen}")
    print(f"[OracleMutate] Candidates: {total_written}")
    print(f"[OracleMutate] Output: {output}")


if __name__ == "__main__":
    main()
