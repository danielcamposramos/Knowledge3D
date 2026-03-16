#!/usr/bin/env bash
# K3D Bare Metal Provisioner
# Installs KDE SparkyLinux + NVIDIA GPU + VSCode + K3D on a fresh bare metal server.
#
# Usage:
#   sudo ./bootstrap.sh              # full install
#   sudo ./bootstrap.sh --phase 3    # run only phase 3 (dev tools)
#   sudo ./bootstrap.sh --dry-run    # show what would be done
#
# Target: Debian 13 (Trixie) / SparkyLinux 9 (Tiamat)
# Provider: MaxCloudON (Ryzen 9 5950X + RTX 3090)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONF_FILE="${SCRIPT_DIR}/provision.conf"
LOG_FILE="/var/log/k3d-provision.log"
DRY_RUN=0
PHASE_ONLY=""

# --- Parse Arguments ---
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)  DRY_RUN=1; shift ;;
        --phase)    PHASE_ONLY="$2"; shift 2 ;;
        --help|-h)
            echo "Usage: sudo $0 [--dry-run] [--phase N]"
            echo "Phases: 1=base-os 2=gpu 3=devtools 4=k3d-runtime 5=daemon"
            exit 0 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

# --- Load Configuration ---
if [[ -f "$CONF_FILE" ]]; then
    # shellcheck source=provision.conf
    source "$CONF_FILE"
else
    echo "WARN: $CONF_FILE not found, using defaults"
    K3D_USER="k3d"
    K3D_HOME="/K3D"
    K3D_REPO_DIR="/K3D/Knowledge3D"
    K3D_LOCAL_DIR="/K3D/Knowledge3D.local"
    K3D_ENVS_DIR="/K3D/Knowledge3D.local/envs"
    K3D_DATASETS_DIR="/K3D/Knowledge3D.local/datasets"
    K3D_REPO_URL="https://github.com/danielcamposramos/Knowledge3D.git"
    K3D_REPO_BRANCH="main"
    GPU_DRIVER_VERSION="550"
    CUDA_VERSION="12.4"
    MINIFORGE_URL="https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh"
    INSTALL_KDE="yes"
    INSTALL_VSCODE="yes"
    K3D_DAEMON_PORT="7333"
    K3D_DAEMON_STORAGE_ROOT="/K3D/Knowledge3D.local"
    ENABLE_DAEMON_SERVICE="yes"
fi

# --- Helpers ---
log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_FILE"; }
run() {
    if [[ $DRY_RUN -eq 1 ]]; then
        log "[DRY-RUN] $*"
    else
        log "[RUN] $*"
        "$@" 2>&1 | tee -a "$LOG_FILE"
    fi
}
should_run_phase() {
    [[ -z "$PHASE_ONLY" ]] || [[ "$PHASE_ONLY" == "$1" ]]
}

# --- Pre-flight Checks ---
if [[ $EUID -ne 0 && $DRY_RUN -eq 0 ]]; then
    echo "ERROR: Run as root (sudo $0)" >&2
    exit 1
fi

mkdir -p "$(dirname "$LOG_FILE")"
log "=== K3D Bare Metal Provisioning Start ==="
log "Config: user=$K3D_USER home=$K3D_HOME gpu_driver=$GPU_DRIVER_VERSION cuda=$CUDA_VERSION"

# ============================================================================
# PHASE 1: Base OS — Repositories + KDE Desktop
# ============================================================================
if should_run_phase 1; then
    log "=== PHASE 1: Base OS ==="

    # Detect if SparkyLinux or plain Debian
    if grep -qi sparky /etc/os-release 2>/dev/null; then
        log "Detected SparkyLinux"
        IS_SPARKY=1
    else
        log "Detected Debian (non-Sparky)"
        IS_SPARKY=0
    fi

    # 1.1 — Ensure non-free firmware components
    log "Phase 1.1: Configuring Debian repositories..."
    run mkdir -p /etc/apt/sources.list.d

    if [[ ! -f /etc/apt/sources.list.d/debian.sources ]]; then
        cat > /etc/apt/sources.list.d/debian.sources <<'DEBSRC'
Types: deb deb-src
URIs: https://deb.debian.org/debian/
Suites: testing
Components: non-free-firmware non-free contrib main
Signed-By: /usr/share/keyrings/debian-archive-keyring.gpg

Types: deb deb-src
URIs: https://deb.debian.org/debian-security/
Suites: testing-security
Components: non-free-firmware non-free contrib main
Signed-By: /usr/share/keyrings/debian-archive-keyring.gpg

Types: deb deb-src
URIs: https://deb.debian.org/debian/
Suites: testing-updates
Components: non-free-firmware non-free contrib main
Signed-By: /usr/share/keyrings/debian-archive-keyring.gpg
DEBSRC
        log "Created debian.sources"
    fi

    # 1.2 — Add SparkyLinux repos (if not already Sparky)
    if [[ $IS_SPARKY -eq 0 && ! -f /etc/apt/sources.list.d/sparky.sources ]]; then
        log "Phase 1.2: Adding SparkyLinux repositories..."
        run wget -qO /etc/apt/trusted.gpg.d/sparky-repo.asc \
            https://repo.sparkylinux.org/sparky.gpg.key

        cat > /etc/apt/sources.list.d/sparky.sources <<'SPARKYSRC'
Types: deb deb-src
URIs: https://repo.sparkylinux.org/
Suites: core
Components: main
Signed-By: /etc/apt/trusted.gpg.d/sparky-repo.asc

Types: deb deb-src
URIs: https://repo.sparkylinux.org/
Suites: tiamat
Components: main
Signed-By: /etc/apt/trusted.gpg.d/sparky-repo.asc
SPARKYSRC
        log "Added Sparky repos"
    fi

    # 1.3 — Add Debian Multimedia
    if [[ ! -f /etc/apt/sources.list.d/debian-multimedia.sources ]]; then
        log "Phase 1.3: Adding Debian Multimedia..."
        run wget -qO /usr/share/keyrings/deb-multimedia-keyring.pgp \
            https://www.deb-multimedia.org/pool/main/d/deb-multimedia-keyring/deb-multimedia-keyring_2016.8.1_all.deb 2>/dev/null || true
        # Install the keyring package properly
        run apt-get update -oAcquire::AllowInsecureRepositories=true -qq 2>/dev/null || true
        run apt-get install -y --allow-unauthenticated deb-multimedia-keyring 2>/dev/null || true

        cat > /etc/apt/sources.list.d/debian-multimedia.sources <<'MMSRC'
Types: deb
URIs: https://www.deb-multimedia.org
Suites: trixie
Components: main non-free
Signed-By: /usr/share/keyrings/deb-multimedia-keyring.pgp
Enabled: yes
MMSRC
        log "Added Multimedia repos"
    fi

    # 1.4 — Update package index
    log "Phase 1.4: Updating package index..."
    run apt-get update -qq

    # 1.5 — Install essential packages
    log "Phase 1.5: Installing base packages..."
    run apt-get install -y --no-install-recommends \
        curl wget git git-lfs ca-certificates gnupg lsb-release \
        build-essential cmake pkg-config \
        tmux htop iotop neofetch tree unzip jq \
        openssh-server ufw \
        firmware-linux firmware-misc-nonfree

    # 1.6 — Install KDE Plasma Desktop
    if [[ "$INSTALL_KDE" == "yes" ]]; then
        log "Phase 1.6: Installing KDE Plasma Desktop..."
        run apt-get install -y --no-install-recommends \
            kde-plasma-desktop sddm sddm-theme-breeze \
            plasma-nm plasma-pa plasma-workspace \
            dolphin konsole kate ark okular gwenview spectacle \
            breeze-icon-theme fonts-noto
        run systemctl enable sddm
        run systemctl set-default graphical.target
        log "KDE Plasma installed, SDDM enabled"
    fi

    # 1.7 — Create K3D system user
    if ! id "$K3D_USER" &>/dev/null; then
        log "Phase 1.7: Creating user $K3D_USER..."
        run useradd -m -s /bin/bash -G sudo,video,render "$K3D_USER"
        log "User $K3D_USER created. Set password with: passwd $K3D_USER"
    fi

    # 1.8 — Create K3D directories
    log "Phase 1.8: Creating K3D directories..."
    run mkdir -p "$K3D_HOME" "$K3D_LOCAL_DIR" "$K3D_ENVS_DIR" "$K3D_DATASETS_DIR"
    run chown -R "$K3D_USER:$K3D_USER" "$K3D_HOME"

    log "=== PHASE 1 COMPLETE ==="
fi

# ============================================================================
# PHASE 2: GPU — NVIDIA Driver + CUDA Toolkit
# ============================================================================
if should_run_phase 2; then
    log "=== PHASE 2: GPU Stack ==="

    # 2.1 — Install NVIDIA driver from Debian non-free
    log "Phase 2.1: Installing NVIDIA driver ${GPU_DRIVER_VERSION}..."
    run apt-get install -y \
        nvidia-driver nvidia-driver-libs \
        nvidia-cuda-toolkit nvidia-cuda-dev \
        nvidia-smi

    # 2.2 — NVIDIA Container Toolkit (for future Docker GPU)
    if [[ ! -f /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg ]]; then
        log "Phase 2.2: Adding NVIDIA Container Toolkit repo..."
        run curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
            | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

        cat > /etc/apt/sources.list.d/nvidia-container-toolkit.sources <<'NVCSRC'
Types: deb
URIs: https://nvidia.github.io/libnvidia-container/stable/deb/amd64/
Suites: /
Components:
Signed-By: /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
NVCSRC
        run apt-get update -qq
        run apt-get install -y nvidia-container-toolkit
    fi

    # 2.3 — Verify GPU
    log "Phase 2.3: Verifying GPU..."
    if command -v nvidia-smi &>/dev/null; then
        nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv | tee -a "$LOG_FILE"
    else
        log "WARN: nvidia-smi not found — reboot may be required"
    fi

    # 2.4 — Blacklist nouveau
    if [[ ! -f /etc/modprobe.d/blacklist-nouveau.conf ]]; then
        log "Phase 2.4: Blacklisting nouveau..."
        cat > /etc/modprobe.d/blacklist-nouveau.conf <<'NOUVEAU'
blacklist nouveau
options nouveau modeset=0
NOUVEAU
        run update-initramfs -u
    fi

    log "=== PHASE 2 COMPLETE (reboot recommended before Phase 3) ==="
fi

# ============================================================================
# PHASE 3: Development Tools — VSCode + AI Extensions
# ============================================================================
if should_run_phase 3; then
    log "=== PHASE 3: Development Tools ==="

    # 3.1 — Install VSCode
    if [[ "$INSTALL_VSCODE" == "yes" ]]; then
        if ! command -v code &>/dev/null; then
            log "Phase 3.1: Installing Visual Studio Code..."
            run wget -qO /usr/share/keyrings/microsoft.gpg \
                https://packages.microsoft.com/keys/microsoft.asc

            cat > /etc/apt/sources.list.d/vscode.sources <<'VSSRC'
Types: deb
URIs: https://packages.microsoft.com/repos/code
Suites: stable
Components: main
Architectures: amd64
Signed-By: /usr/share/keyrings/microsoft.gpg
VSSRC
            run apt-get update -qq
            run apt-get install -y code
        fi

        # 3.2 — Install VSCode extensions (as K3D user)
        log "Phase 3.2: Installing VSCode extensions..."
        K3D_EXTENSIONS=(
            # AI Assistants
            "anthropic.claude-code"
            "saoudrizwan.claude-dev"
            "openai.chatgpt"
            "google.geminicodeassist"
            "google.gemini-cli-vscode-ide-companion"
            # Git
            "eamodio.gitlens"
            "donjayamanne.githistory"
            "github.vscode-pull-request-github"
            # Python
            "ms-python.python"
            "ms-python.vscode-pylance"
            "ms-python.debugpy"
            "ms-python.vscode-python-envs"
            # C++/CUDA
            "ms-vscode.cpptools"
            "ms-vscode.cpptools-extension-pack"
            "ms-vscode.cmake-tools"
            "twxs.cmake"
            # Containers/K8s
            "ms-azuretools.vscode-docker"
            "ms-vscode-remote.remote-containers"
            "ms-kubernetes-tools.vscode-kubernetes-tools"
            # Utilities
            "redhat.vscode-yaml"
            "davidanson.vscode-markdownlint"
            "christian-kohler.npm-intellisense"
        )

        for ext in "${K3D_EXTENSIONS[@]}"; do
            log "  Installing extension: $ext"
            run su - "$K3D_USER" -c "code --install-extension $ext --force" 2>/dev/null || true
        done
    fi

    # 3.3 — Install Docker (for optional containerized deployment)
    if ! command -v docker &>/dev/null; then
        log "Phase 3.3: Installing Docker..."
        run install -m 0755 -d /etc/apt/keyrings
        run curl -fsSL https://download.docker.com/linux/debian/gpg \
            -o /etc/apt/keyrings/docker.asc
        run chmod a+r /etc/apt/keyrings/docker.asc

        cat > /etc/apt/sources.list.d/docker.sources <<'DOCKSRC'
Types: deb
URIs: https://download.docker.com/linux/debian/
Suites: bookworm
Components: stable
Signed-By: /etc/apt/keyrings/docker.asc
Architectures: amd64
DOCKSRC
        run apt-get update -qq
        run apt-get install -y docker-ce docker-ce-cli containerd.io \
            docker-buildx-plugin docker-compose-plugin
        run usermod -aG docker "$K3D_USER"
    fi

    log "=== PHASE 3 COMPLETE ==="
fi

# ============================================================================
# PHASE 4: K3D Runtime — Miniforge + Conda Envs + Repository
# ============================================================================
if should_run_phase 4; then
    log "=== PHASE 4: K3D Runtime ==="

    K3D_USER_HOME=$(eval echo "~${K3D_USER}")
    MINIFORGE_DIR="${K3D_USER_HOME}/miniforge"

    # 4.1 — Install Miniforge
    if [[ ! -d "$MINIFORGE_DIR" ]]; then
        log "Phase 4.1: Installing Miniforge..."
        MINIFORGE_INSTALLER="/tmp/miniforge-installer.sh"
        run wget -qO "$MINIFORGE_INSTALLER" "$MINIFORGE_URL"
        run chmod +x "$MINIFORGE_INSTALLER"
        run su - "$K3D_USER" -c "bash $MINIFORGE_INSTALLER -b -p $MINIFORGE_DIR"
        rm -f "$MINIFORGE_INSTALLER"

        # Initialize conda for the user
        run su - "$K3D_USER" -c "$MINIFORGE_DIR/bin/conda init bash"

        # Configure conda to use SSD env path
        run su - "$K3D_USER" -c "cat >> ~/.condarc <<CONDAEOF
envs_dirs:
  - ${K3D_ENVS_DIR}
  - ${MINIFORGE_DIR}/envs
pkgs_dirs:
  - ${K3D_LOCAL_DIR}/conda-pkgs
  - ${MINIFORGE_DIR}/pkgs
CONDAEOF"
    fi

    # 4.2 — Clone K3D repository
    if [[ ! -d "$K3D_REPO_DIR/.git" ]]; then
        log "Phase 4.2: Cloning K3D repository..."
        run su - "$K3D_USER" -c "git clone --branch $K3D_REPO_BRANCH $K3D_REPO_URL $K3D_REPO_DIR"
    else
        log "Phase 4.2: K3D repository already exists, pulling latest..."
        run su - "$K3D_USER" -c "cd $K3D_REPO_DIR && git pull --ff-only origin $K3D_REPO_BRANCH"
    fi

    # 4.3 — Create conda environments from yml specs
    log "Phase 4.3: Creating conda environments..."

    CONDA_BIN="${MINIFORGE_DIR}/bin/conda"

    # k3d-cranium: primary GPU environment
    if [[ ! -d "${K3D_ENVS_DIR}/k3d-cranium" ]]; then
        log "  Creating k3d-cranium environment..."
        run su - "$K3D_USER" -c \
            "$CONDA_BIN env create -f ${K3D_REPO_DIR}/envs/k3d-cranium.yml -p ${K3D_ENVS_DIR}/k3d-cranium"
    fi

    # k3d-trm: GPU PTX test rig
    if [[ -f "${K3D_REPO_DIR}/envs/k3d-trm.yml" && ! -d "${K3D_ENVS_DIR}/k3d-trm" ]]; then
        log "  Creating k3d-trm environment..."
        run su - "$K3D_USER" -c \
            "$CONDA_BIN env create -f ${K3D_REPO_DIR}/envs/k3d-trm.yml -p ${K3D_ENVS_DIR}/k3d-trm"
    fi

    # 4.4 — Set CUDA_VISIBLE_DEVICES in user profile
    log "Phase 4.4: Configuring GPU environment..."
    K3D_PROFILE="${K3D_USER_HOME}/.bashrc"
    if ! grep -q 'CUDA_VISIBLE_DEVICES' "$K3D_PROFILE" 2>/dev/null; then
        cat >> "$K3D_PROFILE" <<'GPUENV'

# K3D GPU Configuration
# On KDE systems with iGPU, expose the discrete NVIDIA GPU
export CUDA_VISIBLE_DEVICES=0
# K3D paths
export PYTHONPATH="/K3D/Knowledge3D:${PYTHONPATH:-}"
GPUENV
    fi

    log "=== PHASE 4 COMPLETE ==="
fi

# ============================================================================
# PHASE 5: K3D Daemon Service
# ============================================================================
if should_run_phase 5; then
    log "=== PHASE 5: K3D Daemon Service ==="

    K3D_USER_HOME=$(eval echo "~${K3D_USER}")
    MINIFORGE_DIR="${K3D_USER_HOME}/miniforge"
    CONDA_BIN="${MINIFORGE_DIR}/bin/conda"

    # 5.1 — Create systemd service
    log "Phase 5.1: Creating k3d-daemon.service..."
    cat > /etc/systemd/system/k3d-daemon.service <<SVCEOF
[Unit]
Description=K3D Knowledge Universe Daemon
After=network.target nvidia-persistenced.service
Wants=nvidia-persistenced.service

[Service]
Type=simple
User=${K3D_USER}
Group=${K3D_USER}
WorkingDirectory=${K3D_REPO_DIR}
Environment=CUDA_VISIBLE_DEVICES=0
Environment=PYTHONPATH=${K3D_REPO_DIR}
Environment=K3D_STORAGE_ROOT=${K3D_DAEMON_STORAGE_ROOT}
Environment=K3D_DAEMON_PORT=${K3D_DAEMON_PORT}
ExecStart=${CONDA_BIN} run -p ${K3D_ENVS_DIR}/k3d-cranium python -m knowledge3d.daemon.main --port ${K3D_DAEMON_PORT} --storage-root ${K3D_DAEMON_STORAGE_ROOT}
Restart=on-failure
RestartSec=10
StandardOutput=journal
StandardError=journal
SyslogIdentifier=k3d-daemon

# GPU access
SupplementaryGroups=video render

# Resource limits
LimitNOFILE=65536
LimitMEMLOCK=infinity

[Install]
WantedBy=multi-user.target
SVCEOF

    # 5.2 — Create convenience scripts
    log "Phase 5.2: Creating CLI convenience scripts..."

    cat > /usr/local/bin/k3d-client <<'CLIENTEOF'
#!/usr/bin/env bash
# K3D Client — sends queries to the running daemon
CONDA_BIN="${HOME}/miniforge/bin/conda"
K3D_DIR="/K3D/Knowledge3D"
exec "$CONDA_BIN" run -p /K3D/Knowledge3D.local/envs/k3d-cranium \
    env PYTHONPATH="$K3D_DIR" \
    python "$K3D_DIR/scripts/k3d_client.py" "$@"
CLIENTEOF
    chmod +x /usr/local/bin/k3d-client

    cat > /usr/local/bin/k3d-benchmark <<'BENCHEOF'
#!/usr/bin/env bash
# K3D Benchmark Runner
CONDA_BIN="${HOME}/miniforge/bin/conda"
K3D_DIR="/K3D/Knowledge3D"
exec "$CONDA_BIN" run -p /K3D/Knowledge3D.local/envs/k3d-cranium \
    env PYTHONPATH="$K3D_DIR" CUDA_VISIBLE_DEVICES=0 \
    python "$K3D_DIR/scripts/run_all_benchmarks.py" "$@"
BENCHEOF
    chmod +x /usr/local/bin/k3d-benchmark

    # 5.3 — Enable and start service
    if [[ "$ENABLE_DAEMON_SERVICE" == "yes" ]]; then
        log "Phase 5.3: Enabling K3D daemon service..."
        run systemctl daemon-reload
        run systemctl enable k3d-daemon.service
        run systemctl start k3d-daemon.service || log "WARN: Daemon start failed (GPU may need reboot)"
    fi

    # 5.4 — Print status
    log "Phase 5.4: Installation summary"
    log "  K3D User:    $K3D_USER"
    log "  K3D Home:    $K3D_HOME"
    log "  Repository:  $K3D_REPO_DIR"
    log "  Conda Envs:  $K3D_ENVS_DIR"
    log "  Daemon Port: $K3D_DAEMON_PORT"
    log "  Service:     systemctl status k3d-daemon"
    log "  Client:      k3d-client query 'What is 2+3?'"
    log "  Benchmarks:  k3d-benchmark"

    log "=== PHASE 5 COMPLETE ==="
fi

log "=== K3D PROVISIONING COMPLETE ==="
log "Next steps:"
log "  1. Reboot to load NVIDIA driver: sudo reboot"
log "  2. Log in as $K3D_USER via KDE/SDDM"
log "  3. Verify GPU: nvidia-smi"
log "  4. Check daemon: systemctl status k3d-daemon"
log "  5. Run benchmarks: k3d-benchmark"
