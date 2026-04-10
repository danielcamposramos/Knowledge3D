#!/usr/bin/env bash
set -euo pipefail

# K3D environment runner: resolves SSD-hosted env prefixes before falling back to names.
ROOT_DIR=$(cd "$(dirname "$0")/.." && pwd)
K3D_ENV_ROOT=${K3D_ENV_ROOT:-/K3D/Knowledge3D.local/envs}
ENV_SELECTION=${K3D_ENV:-k3d-cranium}

if [[ $# -gt 0 && "$1" == "run" ]]; then
  shift 1
fi

while [[ $# -gt 0 ]]; do
  case "$1" in
    -e|--env)
      ENV_SELECTION="$2"
      shift 2
      ;;
    --env=*)
      ENV_SELECTION="${1#*=}"
      shift 1
      ;;
    --print-env)
      PRINT_ENV=1
      shift 1
      ;;
    --)
      shift 1
      break
      ;;
    *)
      break
      ;;
  esac
done

resolve_env_prefix() {
  local selection="$1"
  if [[ -n "$selection" && -x "$selection/bin/python" ]]; then
    printf '%s\n' "$selection"
    return 0
  fi
  if [[ -n "$selection" && -x "$K3D_ENV_ROOT/$selection/bin/python" ]]; then
    printf '%s\n' "$K3D_ENV_ROOT/$selection"
    return 0
  fi
  return 1
}

ENV_PREFIX=""
if ENV_PREFIX=$(resolve_env_prefix "$ENV_SELECTION"); then
  :
fi

activate_conda() {
  if [ -f "$HOME/miniforge/etc/profile.d/conda.sh" ]; then
    # shellcheck disable=SC1090
    . "$HOME/miniforge/etc/profile.d/conda.sh"
    return 0
  fi
  if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    # shellcheck disable=SC1090
    . "$HOME/miniconda3/etc/profile.d/conda.sh"
    return 0
  fi
  return 1
}

if activate_conda; then
  if [[ -n "$ENV_PREFIX" ]]; then
    conda activate "$ENV_PREFIX" >/dev/null 2>&1 || true
  else
    conda activate "$ENV_SELECTION" >/dev/null 2>&1 || true
  fi
fi

# Prefix-path activation still works when conda metadata is unavailable.
if [[ -n "$ENV_PREFIX" ]]; then
  export PATH="$ENV_PREFIX/bin:$PATH"
elif [ -x "$ROOT_DIR/.venv_k3dml/bin/python" ]; then
  export PATH="$ROOT_DIR/.venv_k3dml/bin:$PATH"
fi

# Common env
export PYTHONPATH="$ROOT_DIR:${PYTHONPATH:-}"
export K3D_PTX_STRICT=${K3D_PTX_STRICT:-1}
export K3D_FORCE_PTX_FUSE=${K3D_FORCE_PTX_FUSE:-1}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

# Kaggle API token — loaded from local secrets, never committed to repo
_KAGGLE_KEY_FILE="/K3D/Knowledge3D.local/secrets/kaggle_api_key.txt"
if [[ -z "${KAGGLE_API_TOKEN:-}" && -r "$_KAGGLE_KEY_FILE" ]]; then
  export KAGGLE_API_TOKEN
  KAGGLE_API_TOKEN=$(tr -d '[:space:]' < "$_KAGGLE_KEY_FILE")
fi
unset _KAGGLE_KEY_FILE

if [[ "${PRINT_ENV:-0}" == "1" ]]; then
  if [[ -n "$ENV_PREFIX" ]]; then
    printf '%s\n' "$ENV_PREFIX"
  else
    printf '%s\n' "$ENV_SELECTION"
  fi
  exit 0
fi

exec "$@"
