#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
K3D_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
LOCAL_PYENV_ROOT="$K3D_DIR/.pyenv"
VENV_NAME="k3d-venv"
MIN_PYTHON_VERSION="3.10"
MIN_BUN_VERSION="1.3.9"

echo "=== Knowledge3D Installation Script ==="
echo "Project directory: $K3D_DIR"
echo "Demo directory: $SCRIPT_DIR"

# Check if system python3 exists and validate version
if ! command -v python3 &> /dev/null; then
    echo "ERROR: python3 is not installed."
    exit 1
fi

PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
echo "Detected Python version: $PYTHON_VERSION"

# Compare versions (convert to comparable integers: 3.10 -> 310, 3.9 -> 309)
PYTHON_VER_NUM=$(echo "$PYTHON_VERSION" | awk -F. '{printf "%d%02d", $1, $2}')
MIN_PYTHON_VER_NUM=$(echo "$MIN_PYTHON_VERSION" | awk -F. '{printf "%d%02d", $1, $2}')

if [ "$PYTHON_VER_NUM" -lt "$MIN_PYTHON_VER_NUM" ]; then
    echo "ERROR: Python $MIN_PYTHON_VERSION or higher is required (found $PYTHON_VERSION)"
    exit 1
fi

# Check if bun exists and validate version
if ! command -v bun &> /dev/null; then
    echo "ERROR: bun is not installed."
    exit 1
fi

BUN_VERSION=$(bun --version)
echo "Detected Bun version: $BUN_VERSION"

# Compare bun versions
BUN_VER_NUM=$(echo "$BUN_VERSION" | awk -F. '{printf "%d%03d%03d", $1, $2, $3}')
MIN_BUN_VER_NUM=$(echo "$MIN_BUN_VERSION" | awk -F. '{printf "%d%03d%03d", $1, $2, $3}')

if [ "$BUN_VER_NUM" -lt "$MIN_BUN_VER_NUM" ]; then
    echo "ERROR: Bun $MIN_BUN_VERSION or higher is required (found $BUN_VERSION)"
    exit 1
fi

export BUN_VERSION

# Create local virtualenv if it doesn't exist
if [ ! -d "$LOCAL_PYENV_ROOT" ]; then
    echo "Creating local virtualenv: $LOCAL_PYENV_ROOT"
    python3 -m venv --copies "$LOCAL_PYENV_ROOT"
else
    echo "Local virtualenv already exists: $LOCAL_PYENV_ROOT"
fi

# Activate the local virtualenv
echo "Activating local virtualenv..."
source "$LOCAL_PYENV_ROOT/bin/activate"

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install Knowledge3D in editable mode
echo "Installing Knowledge3D..."
pip install -e "$K3D_DIR"

# Install additional requirements
if [ -f "$K3D_DIR/requirements.txt" ]; then
    echo "Installing additional requirements..."
    pip install -r "$K3D_DIR/requirements.txt"
fi

# Install websockets for bridge server
pip install websockets

# Install viewer dependencies
echo "Installing viewer dependencies..."
(cd "$K3D_DIR/viewer" && bun install)

echo ""
echo "=== Installation Complete ==="
echo "Virtualenv: $LOCAL_PYENV_ROOT"
echo "Python: $(python --version)"
echo "Data directory: $DATA_DIR"
echo ""
echo "To run the demo:"
echo "  bun run dev"
