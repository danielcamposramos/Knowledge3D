#!/usr/bin/env python3
"""Build final synthesis report for Christoph's Sovereign Systems Charter request."""

from __future__ import annotations

import json
from pathlib import Path


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _md_link(title: str, url: str) -> str:
    safe = title.replace("[", "(").replace("]", ")")
    return f"[{safe}]({url})"


def _render_top_list(items: list[dict], limit: int = 20) -> str:
    lines: list[str] = []
    for idx, item in enumerate(items[:limit], start=1):
        lines.append(
            f"{idx}. {_md_link(item['title'], item['url'])} "
            f"(score={item.get('score')}, pos={item.get('pos')}, neg={item.get('neg')})"
        )
    return "\n".join(lines) if lines else "1. (none)"


def main() -> int:
    root = Path("docs/Sovereign_Systems_Charter")
    rel = _load_json(root / "relevance_partition.json")
    counts = rel["counts"]
    core = rel.get("core_top", [])
    supporting = rel.get("supporting_top", [])
    ignored = rel.get("ignored_sample", [])

    selected = int(counts["core"]) + int(counts["supporting"])
    selected_pct = (selected / int(counts["total"])) * 100.0 if counts["total"] else 0.0

    report = root / "FINAL_REPORT_FOR_CHRISTOPH.md"
    report.write_text(
        "\n".join(
            [
                "# Final Synthesis Report: Sovereign Systems Charter Mission",
                "",
                "## Context",
                "This report answers Christoph's request to read the author's corpus around the",
                "Sovereign Systems Charter and synthesize a practical perspective for building",
                "LLM systems responsibly, with explicit boundary logic.",
                "",
                "Source charter reference:",
                "- https://hblazer.substack.com/p/the-sovereign-systems-charter",
                "",
                "## What We Did (Claude-Style Scope Discipline)",
                "1. Collected full corpus from the completed scrape.",
                "2. Generated per-post text and extractive summaries.",
                "3. Applied a relevance partition aligned to mission topics:",
                "   sovereignty, boundaries, charter principles, privacy/transparency,",
                "   LLM responsibility, governance, ethics, and human agency.",
                "4. Built this synthesis from the selected subset and ignored off-mission drift.",
                "",
                "## Corpus Accounting",
                f"- Total posts processed: **{counts['total']}**",
                f"- Mission-core: **{counts['core']}**",
                f"- Mission-supporting: **{counts['supporting']}**",
                f"- Ignored (off-mission/noise): **{counts['ignore']}**",
                f"- Selected for synthesis: **{selected} / {counts['total']} ({selected_pct:.1f}%)**",
                "",
                "## What To Do (Actionable Synthesis)",
                "1. Treat boundaries as first-class system primitives.",
                "   This means each capability in PM-KR/K3D must declare:",
                "   hard boundaries (forbidden), soft boundaries (override with consequence),",
                "   and ambiguity boundaries (requires clarification step).",
                "2. Encode privacy/transparency as a controllable boundary, not an absolute.",
                "   Keep Christoph's point explicit: radical transparency is unsafe;",
                "   hidden structures are sometimes necessary. Implement explainability with",
                "   selective disclosure and role-based access.",
                "3. Shift from opinion-level governance to contract-level governance.",
                "   For each high-impact action, bind: provenance, intent, rule applied,",
                "   and remediation path if boundary is crossed.",
                "4. Keep humans as final authority for cross-boundary actions.",
                "   The model can propose, simulate, and warn; the human confirms",
                "   when boundaries involve legal, ethical, social, or physical risk.",
                "5. Make sovereignty operational, not rhetorical.",
                "   Define measurable invariants:",
                "   local control of execution path, auditable decisions, and bounded external dependencies.",
                "6. Make LLM usage bounded by architecture contracts.",
                "   LLMs should operate as constrained components under boundary policies,",
                "   never as opaque ultimate arbiters.",
                "7. Build fractal boundary governance.",
                "   Apply boundary rules consistently across layers:",
                "   token/program, module, workflow, user-space, and cross-system interfaces.",
                "",
                "## What To Ignore (Deliberate Exclusions)",
                "1. Topic drift not tied to boundary architecture:",
                "   geopolitics, personalities, event commentary, and market speculation.",
                "2. Claims without operational translation.",
                "   If a claim cannot be converted into a testable system requirement,",
                "   it stays out of engineering scope.",
                "3. Absolutist framing that collapses nuance.",
                "   Especially transparency absolutism and single-axis moral narratives",
                "   that do not model trade-offs.",
                "4. High-noise rhetorical repetition.",
                "   Repeated posts with no new boundary construct were deprioritized.",
                "",
                "## PM-KR -> K3D Implementation Mapping",
                "1. Boundary Contract Schema (new):",
                "   each procedural entry carries `boundary_type`, `crossing_conditions`,",
                "   `required_authority`, `audit_trace`, `remediation_rule`.",
                "2. Boundary Decision Gate in runtime:",
                "   before execution, evaluate boundary contract and require escalation",
                "   when crossing hard/critical boundaries.",
                "3. Privacy/Transparency Dial:",
                "   add policy levels for what the system reveals in explanations",
                "   (public, collaborator, regulator, internal).",
                "4. Responsible LLM Envelope:",
                "   force LLM outputs through boundary validators before they can mutate memory/state.",
                "",
                "## Priority Reading Set (Core)",
                _render_top_list(core, limit=25),
                "",
                "## Supporting Reading Set",
                _render_top_list(supporting, limit=20),
                "",
                "## Ignored Sample (For Traceability)",
                _render_top_list(ignored, limit=20),
                "",
                "## Deliverables in This Folder",
                "- `posts_full/*.txt`: one-by-one normalized texts",
                "- `posts_full/*.md`: one-by-one summaries",
                "- `manifest.json`: full index",
                "- `relevance_partition.json`: mission partition data",
                "- `FINAL_REPORT_FOR_CHRISTOPH.md`: this synthesis",
                "",
                "## Recommendation to Send Christoph",
                "Send this report with two commitments:",
                "1. We will codify boundary contracts in PM-KR vocabulary and reference implementation.",
                "2. We will explicitly model the privacy/transparency boundary as a tunable policy layer,",
                "   not as ideology.",
                "",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"[done] report={report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
