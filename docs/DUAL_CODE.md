# Dual Code Paradigm: Human‑Readable (HR) and Machine‑Runtime (MR)

K3D maintains two representations of source when running locally:

- HR (Human‑Readable): idiomatic, commented, richly documented code for humans and AI to study.
- MR (Machine‑Runtime): stripped, compact, comment‑free output optimized for loading and execution (no semantics change).

This is achieved by a simple optimizer that copies sources into a sibling folder outside the repo and applies safe transforms:

- Python: remove comments, remove docstrings (inserting `pass` if needed), compress blank lines.
- JS/TS: remove `//` and `/* */` comments (respecting strings/template literals), compress blank lines.
- Preserve indentation and token order; do not change semantics.

## Tooling

CLI: `codeopt`

Examples:

```bash
# Produce MR output for the whole repo into ../Knowledge3D.local/mr
codeopt --in . --out ../Knowledge3D.local/mr --lang auto --stats

# Only Python files
codeopt --in k3dgen knowledge3d --out ../Knowledge3D.local/mr --lang py --stats
```

Output layout mirrors input directories under the output root. This folder is not tracked by git.

## Guarantees

- Python MR files compile; docstring‑only bodies get `pass`.
- JS/TS comment removal is string‑aware; block comments are removed; multiline string content is preserved.
- Reports bytes saved and a per‑file summary when `--stats` is enabled.

