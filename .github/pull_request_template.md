## Summary

Describe what this pull request changes and why. Link to any related issues or TEMP/session notes.

- Related issues: 
- Related docs (e.g., `docs/ROADMAP.md`, `CLAUDE.md`, `SOVEREIGN_SWARM_BRIEFING.md`):

## Type of Change

Select all that apply:

- [ ] Bug fix
- [ ] New feature
- [ ] Performance improvement
- [ ] Documentation update
- [ ] Refactor / internal change
- [ ] Tests / CI only

## Architecture Impact

Check any areas this PR touches and briefly explain the impact.

- [ ] Cranium (PTX kernels / bridges)
- [ ] Galaxy (RAM, k‑NN, resonance)
- [ ] House (GLB, consolidation, tablet)
- [ ] Viewer / Tablet UI
- [ ] Ingestion / training tools
- [ ] Other:

Explanation:

## Implementation Notes

- Key files changed:
- How this reuses or extends existing kernels / bridges:
- Any deviations from standard ENV / sovereignty rules (if any, justify explicitly):

## Testing

Describe how you tested your changes.

- [ ] `pytest -q`
- [ ] Viewer tests (e.g., Jest)
- [ ] Kernel‑specific tests or benchmarks
- [ ] Manual testing

Details:

## Backwards Compatibility / Migration

- Does this change alter public APIs, file formats (e.g., GLB extras), or memory behavior?
- Are there any migration steps or one‑time scripts required?

## Checklist

- [ ] I have read `CONTRIBUTING.md`.
- [ ] I have read and will follow `CODE_OF_CONDUCT.md`.
- [ ] I have updated documentation for this change where appropriate.
- [ ] I have added or updated tests as needed.
- [ ] I have considered memory and GPU usage and kept within project constraints.

