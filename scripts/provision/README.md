# K3D Bare Metal Provisioning

Self-installing provisioner for K3D on GPU bare metal servers.

## Target Providers

| Provider | Plan | CPU | GPU | Status |
|----------|------|-----|-----|--------|
| MaxCloudON | MAX | Ryzen 9 5950X | RTX 3090 | Primary target |

## Usage

### One-Line Bootstrap (from fresh Debian/SparkLinux install)

```bash
curl -fsSL https://raw.githubusercontent.com/danielcamposramos/Knowledge3D/main/scripts/provision/bootstrap.sh | sudo bash
```

### Manual (step by step)

```bash
git clone https://github.com/danielcamposramos/Knowledge3D.git /opt/k3d
cd /opt/k3d/scripts/provision
sudo ./bootstrap.sh
```

### What It Does

1. **Phase 1 — Base OS**: Adds Sparky/Debian repos, installs KDE Plasma desktop
2. **Phase 2 — GPU**: Installs NVIDIA driver 550 + CUDA 12.4 toolkit
3. **Phase 3 — Development**: VSCode + AI extensions (Claude, Codex, Gemini)
4. **Phase 4 — K3D Runtime**: Miniforge + conda envs + K3D repo + daemon service
5. **Phase 5 — Live Service**: Enables and starts `k3d-daemon.service`

### Configuration

Edit `provision.conf` before running to customize:
- `K3D_USER` — system user for K3D (default: `k3d`)
- `K3D_HOME` — K3D data directory (default: `/K3D`)
- `K3D_REPO` — GitHub repo URL
- `GPU_DRIVER_VERSION` — NVIDIA driver branch (default: `550`)
- `CUDA_VERSION` — CUDA toolkit version (default: `12.4`)

### Post-Install

- KDE session available at next login (SDDM display manager)
- VSCode accessible via desktop or `code` CLI
- K3D daemon runs as systemd service: `systemctl status k3d-daemon`
- Query K3D: `k3d-client query "What is 2+3?"`

### Requirements

- Fresh Debian 13 (Trixie) / SparkyLinux 9 (Tiamat) or compatible
- Bare metal with NVIDIA GPU (RTX 3070/3080/3090/4090)
- Root access
- Internet connection
