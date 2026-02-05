"""
Serialize Log Galaxy traces into a VRAM-friendly binary layout.

Binary layout (little-endian, contiguous sections):
  [header (8 bytes magic)]
  trace_offsets: uint32 (N+1)
  step_rule_ids: uint16 (total_steps)
  step_kind: uint8 (total_steps)
  problem_embeddings: float32 (N * embedding_dim)
  trace_result: float32 (N)
  trace_success: uint8 (N)

Metadata JSON stores offsets, counts, and rule registry.
"""

from __future__ import annotations

from array import array
import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


STEP_KIND = {
    "decompose": 1,
    "base": 2,
    "result": 3,
}


def _iter_entries(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def serialize_log_galaxy(
    *,
    jsonl_path: str,
    output_prefix: str,
) -> Tuple[Path, Path]:
    jsonl = Path(jsonl_path)
    if not jsonl.exists():
        raise FileNotFoundError(f"Log galaxy file not found: {jsonl}")

    rule_registry: Dict[str, int] = {}
    rule_list: List[str] = []
    trace_ids: List[str] = []

    trace_offsets = array("I", [0])
    step_rule_ids = array("H")
    step_kind = array("B")
    embeddings = array("f")
    trace_result = array("f")
    trace_success = array("B")

    embedding_dim = None

    for entry in _iter_entries(jsonl):
        trace_ids.append(str(entry.get("trace_id") or ""))
        steps = entry.get("step_sequence", []) or []
        trace_offsets.append(trace_offsets[-1] + len(steps))

        for step in steps:
            rule = step.get("rule") or step.get("label") or "unknown"
            if rule not in rule_registry:
                rule_registry[rule] = len(rule_list)
                rule_list.append(rule)
            step_rule_ids.append(rule_registry[rule])

            kind = str(step.get("kind") or "").lower()
            step_kind.append(STEP_KIND.get(kind, 0))

        embedding = entry.get("problem_embedding") or []
        if embedding_dim is None:
            embedding_dim = len(embedding)
        elif len(embedding) != embedding_dim:
            raise ValueError("Inconsistent embedding dimensions in log galaxy entries.")
        embeddings.extend(float(x) for x in embedding)

        result = entry.get("result")
        trace_result.append(float(result) if result is not None else 0.0)
        trace_success.append(1 if entry.get("success") else 0)

    if embedding_dim is None:
        raise ValueError("No embeddings found in log galaxy.")

    output_prefix_path = Path(output_prefix)
    bin_path = output_prefix_path.with_suffix(".bin")
    meta_path = output_prefix_path.with_suffix(".json")

    offsets: Dict[str, int] = {}
    lengths: Dict[str, int] = {}
    magic = b"K3DLOG1\0"

    with bin_path.open("wb") as f:
        f.write(magic)
        offsets["trace_offsets"] = f.tell()
        f.write(trace_offsets.tobytes())
        lengths["trace_offsets"] = len(trace_offsets)

        offsets["step_rule_ids"] = f.tell()
        f.write(step_rule_ids.tobytes())
        lengths["step_rule_ids"] = len(step_rule_ids)

        offsets["step_kind"] = f.tell()
        f.write(step_kind.tobytes())
        lengths["step_kind"] = len(step_kind)

        offsets["problem_embeddings"] = f.tell()
        f.write(embeddings.tobytes())
        lengths["problem_embeddings"] = len(embeddings)

        offsets["trace_result"] = f.tell()
        f.write(trace_result.tobytes())
        lengths["trace_result"] = len(trace_result)

        offsets["trace_success"] = f.tell()
        f.write(trace_success.tobytes())
        lengths["trace_success"] = len(trace_success)

    metadata = {
        "version": 1,
        "endian": "little",
        "magic": magic.decode("ascii", errors="ignore"),
        "counts": {
            "traces": len(trace_ids),
            "steps": len(step_rule_ids),
            "embedding_dim": int(embedding_dim),
        },
        "offsets": offsets,
        "lengths": lengths,
        "dtypes": {
            "trace_offsets": "uint32",
            "step_rule_ids": "uint16",
            "step_kind": "uint8",
            "problem_embeddings": "float32",
            "trace_result": "float32",
            "trace_success": "uint8",
        },
        "rule_registry": rule_list,
        "trace_ids": trace_ids,
        "step_kind_enum": STEP_KIND,
        "source_jsonl": str(jsonl),
    }

    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return bin_path, meta_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Serialize Log Galaxy JSONL into binary format.")
    parser.add_argument("--jsonl", required=True, help="Path to log galaxy JSONL.")
    parser.add_argument("--output-prefix", required=True, help="Output prefix for .bin/.json.")
    args = parser.parse_args()

    bin_path, meta_path = serialize_log_galaxy(jsonl_path=args.jsonl, output_prefix=args.output_prefix)
    print(f"Wrote {bin_path} and {meta_path}")


if __name__ == "__main__":
    main()
