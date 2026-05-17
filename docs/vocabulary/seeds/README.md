# Galaxy Star Seed Files

## What These Files Are

Seed JSONL files are **Galaxy star payloads consumed at boot by the ingestion
path**. Each line is one JSON object describing a single Galaxy star entry that
should be resident in VRAM for the duration of a live session.

Seed files are the authoritative source for "factory-installed" stars — stars
that must exist before any query can be processed.

## Where They Load

The ingestion-path Galaxy loader at
`knowledge3d/knowledgeverse/galaxy_loader.py` scans directories for `*.jsonl`
files at boot time. It infers the star type from the filename stem and upserts
each line into the VRAM Galaxy table.

**To include this directory in a load pass**, point `_iter_disk_jsonl_paths`
(line 32 of `galaxy_loader.py`) at `docs/vocabulary/seeds/` or copy the seed
files to the data directory the loader already scans.

**NOTE — loader cut pending:** As of 2026-04-18 the loader does not yet
automatically discover `docs/vocabulary/seeds/`. Wiring this path is the next
Codex cut (see `TEMP/CLAUDE_CODEX_GPU_GAME_LOOP_CLOSURE_04.18.2026.md` §10,
checklist item G1).

## The Ingestion-Only Invariant

Seed files are **ingestion-path only**. They are:

- Read once at Galaxy load time to populate VRAM.
- Never read during query-time inference (the sovereign hot path reads VRAM,
  not disk).
- Safe to modify between sessions; changes take effect on the next Galaxy load.

No code under `knowledge3d/cranium/` or any PTX kernel may open or reference
these files. If you find such a reference, it is a sovereignty violation.

## Current Seeds

| File | Stars | Paradigms |
| --- | --- | --- |
| `wine_contracts_seed.jsonl` | 3 | DOM `<p>` (0x01), ARC3 frame (0x02), stdin/stdout text (0x03) |

## Adding New Seeds

1. Create a new `*.jsonl` file in this directory.
2. Register the star kind in `docs/vocabulary/CANONICAL_REGISTRY_SPECIFICATION.md`.
3. If symbolic RPN addresses are used, add `## TODO` markers so the loader can
   enforce address resolution at load time.
4. Re-run the Galaxy loader; verify the star appears in the VRAM table before
   claiming the seed is live.
