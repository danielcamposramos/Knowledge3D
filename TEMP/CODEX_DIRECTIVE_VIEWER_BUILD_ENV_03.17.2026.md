# Codex Directive: Viewer Build Environment

**Date:** March 17, 2026
**Priority:** Must-read before any viewer TypeScript work

---

## Problem

The repo lives on a CIFS/SMB network mount (`/mnt/arquivos/`). npm cannot function there:
- `node_modules` creates ghost directory entries that can't be removed
- Rollup native binaries (`@rollup/rollup-linux-x64-gnu`) fail with `ERR_DLOPEN_FAILED`
- VS Code's TypeScript server holds file handles, preventing cleanup
- Symlinks are unreliable on CIFS

**Do NOT run `npm install`, `npm run build`, `npx vite build`, or `npx jest` from `viewer/`.**

---

## Solution

A dedicated build environment exists on the local SSD:

```
/K3D/Knowledge3D.local/envs/viewer-build/
```

This directory has its own `node_modules`, `package.json`, and build tooling. Source files are synced from the network share via rsync.

---

## How to Build

```bash
# Full pipeline: sync → test → typecheck → build → copy dist/ back to NAS
bash /K3D/Knowledge3D.local/envs/viewer-build/build.sh

# Just tests
bash /K3D/Knowledge3D.local/envs/viewer-build/build.sh --test

# Just build (skip tests)
bash /K3D/Knowledge3D.local/envs/viewer-build/build.sh --build

# Dev server with hot reload
bash /K3D/Knowledge3D.local/envs/viewer-build/build.sh --dev
```

The script automatically:
1. rsync's `src/`, `public/`, `tests/`, and config files from `viewer/` to SSD
2. Installs deps if `package.json` changed
3. Runs the requested operation
4. rsync's `dist/` back to `viewer/dist/` (for `--build` and `--full`)

---

## When Editing Viewer TypeScript

1. **Edit files in** `viewer/src/` on the network share (where git tracks them)
2. **Build/test via** `bash /K3D/Knowledge3D.local/envs/viewer-build/build.sh`
3. **Output appears at** `viewer/dist/` (synced back automatically)

If you add new npm dependencies:
```bash
cd /K3D/Knowledge3D.local/envs/viewer-build
npm install <package> --save        # or --save-dev
cp package.json "/mnt/arquivos/EchoSystems AI Studios/Knowledge 3D Standard/GitHub/Knowledge3D/viewer/package.json"
```

---

## Current State (Verified)

- Node.js v22.22.0, npm 9.2.0
- Vite 7.3.1, Three.js 0.179.1, TypeScript 5.8.3
- `tsc --noEmit` → clean
- `jest` → 5 suites, 8 tests passed
- `vite build` → 723KB bundle (189KB gzipped), built in ~2s
- `@rollup/rollup-linux-x64-gnu` installed and functional

---

## Environment Map

| Environment | Path | Purpose |
|-------------|------|---------|
| `k3d-cranium` | conda on SSD | GPU/PTX kernels, Python tests |
| `k3d-testing` | conda on SSD | CPU-only mock testing |
| `viewer-build` | npm on SSD | Viewer TypeScript/Vite builds |

All build environments live on SSD (`/K3D/Knowledge3D.local/envs/`). Source code lives on NAS (`/mnt/arquivos/`). This separation is intentional — fast compute on local storage, shared source on network storage.
