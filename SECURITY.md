# Security Policy

Knowledge3D is a sovereign, GPU‑native research system under active development. We take security seriously and appreciate responsible reports that help protect users and contributors.

## Supported Versions

This project is primarily developed on the main branch. At this time, only:

- The latest commit on the `main` (or default) branch

is considered supported for security updates. Older tags or forks may not receive fixes.

## Reporting a Vulnerability

If you believe you have found a security vulnerability in the Knowledge3D codebase, infrastructure, or documentation:

- **Do not** open a public GitHub issue for sensitive security reports.
- Instead, please contact the maintainers privately at:

- **Email**: `contact@echosystems.ai`

When reporting, please include (as applicable):

- A clear description of the issue and its potential impact.
- Steps to reproduce (including any scripts or test data, if possible).
- Affected components (e.g., Cranium kernels, sovereign loader, viewer, bridge server).
- Environment details (OS, GPU model, CUDA version, Python version, conda env).

We aim to:

- Acknowledge receipt of your report within a reasonable time.
- Investigate and validate the issue.
- Work with you on mitigation, fixes, and coordinated disclosure when appropriate.

## Scope

In scope:

- Sovereign GPU runtime (`knowledge3d/cranium/**`), including kernels, bridges, and loaders.
- Live server and bridge (`knowledge3d/bridge/**`) and related tools.
- Viewer and tablet components (`viewer/**`) to the extent they expose network or file interfaces.
- Training and ingestion tools that interact with external data or services (`knowledge3d/tools/**`, `knowledge3d/ingestion/**`, `scripts/**`).

Out of scope (for security reporting, but still welcome as regular issues):

- Documentation typos or minor content issues.
- Local development environment misconfigurations not caused by the project.
- Experimental or archived code under `Old_Attempts/` unless clearly used in the active runtime.

## Responsible Disclosure

We kindly request that you:

- Give maintainers a reasonable time to investigate and address the issue before any public disclosure.
- Avoid intentionally accessing, modifying, or destroying data you do not own.
- Follow applicable laws and regulations during your research.

We greatly appreciate responsible security research that helps make Knowledge3D safer for everyone.

