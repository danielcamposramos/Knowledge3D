#!/usr/bin/env python3
"""
Analyze Run 036 diagnostics to understand regression.

Usage:
    python scripts/analyze_run_diagnostics.py /tmp/arc_run_036.log
"""

from __future__ import annotations

import re
import sys
from collections import defaultdict
from pathlib import Path

def analyze_log(log_path: Path) -> None:
    """Extract diagnostic metrics from training log."""

    log_text = log_path.read_text(encoding="utf-8")

    epoch_accuracies = []
    pattern = re.compile(r"Epoch (\d+) \(cycle .*?\): (\d+)/(\d+) correct")
    for match in pattern.finditer(log_text):
        epoch_num = int(match.group(1))
        correct = int(match.group(2))
        total = int(match.group(3))
        epoch_accuracies.append((epoch_num, correct, total, correct / total if total else 0.0))

    vocab_blocks = re.findall(r"\[VOCAB QUALITY Epoch (\d+)\](.*?)(?=\n\[|$)", log_text, re.DOTALL)

    scale_inv_usage = defaultdict(int)
    for match in re.finditer(r"(REL_LINE|REL_RECT|PROP_GRID|FLOOD_REL)", log_text):
        scale_inv_usage[match.group(1)] += 1

    attractors = [int(match.group(1)) for match in re.finditer(r"\[ATTRACTORS?\].*?discovered (\d+)×", log_text)]

    shadow_counts = [int(m) for m in re.findall(r"Shadow entries: (\\d+)", log_text)]

    print("\n" + "=" * 60)
    print("RUN 036 DIAGNOSTIC REPORT")
    print("=" * 60)

    print("\n[ACCURACY ANALYSIS]")
    if epoch_accuracies:
        accs = [item[3] for item in epoch_accuracies]
        best = max(epoch_accuracies, key=lambda item: item[3])
        worst = min(epoch_accuracies, key=lambda item: item[3])
        mean_acc = sum(accs) / len(accs)
        variance = sum((a - mean_acc) ** 2 for a in accs) / len(accs)
        std_dev = variance ** 0.5
        print(f"  Total epochs: {len(epoch_accuracies)}")
        print(f"  Best epoch: {best[1]}/{best[2]} ({best[3] * 100:.1f}%)")
        print(f"  Worst epoch: {worst[1]}/{worst[2]} ({worst[3] * 100:.1f}%)")
        print(f"  Mean accuracy: {mean_acc * 100:.1f}%")
        print(f"  Std deviation: {std_dev * 100:.1f}%")
        mid = len(accs) // 2
        first_half = sum(accs[:mid]) / mid if mid else 0.0
        second_half = sum(accs[mid:]) / (len(accs) - mid) if len(accs) - mid else 0.0
        trend = "↑ Improving" if second_half > first_half else "↓ Declining" if second_half < first_half else "→ Stable"
        print(f"\n  First half avg: {first_half * 100:.1f}%")
        print(f"  Second half avg: {second_half * 100:.1f}%")
        print(f"  Trend: {trend}")
    else:
        print("  ⚠️ No epoch accuracy records found")

    print(f"\n[SCALE-INVARIANT PRIMITIVES USAGE]")
    if scale_inv_usage:
        for prim, count in sorted(scale_inv_usage.items(), key=lambda item: item[1], reverse=True):
            print(f"  {prim}: {count} occurrences")
    else:
        print("  ⚠️ NO scale-invariant primitives detected in log")

    print(f"\n[ATTRACTOR ANALYSIS]")
    if attractors:
        print(f"  Strong attractors detected: {len(attractors)}")
        print(f"  Max attractor strength: {max(attractors)}×")
        print(f"  Mean attractor strength: {sum(attractors) / len(attractors):.1f}×")
        print(f"  Attractors ≥20×: {len([a for a in attractors if a >= 20])}")
    else:
        print("  No strong attractors (≥15×) detected")

    print(f"\n[SHADOW COPY GROWTH]")
    if shadow_counts:
        print(f"  Initial: {shadow_counts[0]} entries")
        print(f"  Final: {shadow_counts[-1]} entries")
        print(f"  Net growth: {shadow_counts[-1] - shadow_counts[0]} entries")
        if len(shadow_counts) > 1:
            print(f"  Growth rate: {(shadow_counts[-1] - shadow_counts[0]) / (len(shadow_counts) - 1):.2f} entries/checkpoint")
    else:
        print("  No shadow entry counts recorded")

    print(f"\n[VOCABULARY QUALITY BLOCKS]")
    print(f"  Total vocab blocks: {len(vocab_blocks)}")
    if vocab_blocks:
        print(f"  First block: Epoch {vocab_blocks[0][0]}")
        print(f"  Last block: Epoch {vocab_blocks[-1][0]}")
        last_block = vocab_blocks[-1][1]
        if "no grammar usage" in last_block.lower():
            print("  ⚠️ Last block shows NO grammar rule usage")
        else:
            print("  ✅ Grammar rules recorded in latest block")

    print("\n" + "=" * 60)
    print("\n[DECISION CRITERIA ASSESSMENT]")
    if epoch_accuracies:
        mean_acc = sum(item[3] for item in epoch_accuracies) / len(epoch_accuracies)
        mean_percent = mean_acc * 100
        print(f"  Run 036 mean accuracy: {mean_percent:.1f}%")
        if mean_percent >= 40:
            print("  ✅ RECOVERED: Continue training (Run 037-039)")
        elif mean_percent >= 35:
            print("  ⚠️ PARTIAL RECOVERY: Run 037 to confirm trend")
        else:
            print("  🚨 PERSISTENT REGRESSION: Stop and investigate")
    else:
        print("  Unable to evaluate accuracy criteria")
    print("=" * 60 + "\n")


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python scripts/analyze_run_diagnostics.py /tmp/arc_run_036.log")
        raise SystemExit(1)

    log_path = Path(sys.argv[1])
    if not log_path.exists():
        print(f"Log file not found: {log_path}")
        raise SystemExit(1)

    analyze_log(log_path)


if __name__ == "__main__":
    main()
