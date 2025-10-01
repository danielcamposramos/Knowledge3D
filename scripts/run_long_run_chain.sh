#!/usr/bin/env bash
set -euo pipefail
cd "$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH=.
export K3D_PTX_STRICT=${K3D_PTX_STRICT:-1}
export K3D_FORCE_PTX_FUSE=${K3D_FORCE_PTX_FUSE:-1}
export K3D_RPN_BEAM=${K3D_RPN_BEAM:-1}
export K3D_RPN_BEAM_WIDTH=${K3D_RPN_BEAM_WIDTH:-5}

scripts/k3d_env.sh run python -m knowledge3d.tools.phase25.long_run \
  --epochs 50 --limit 300 --eval-every 5 \
  --dims "64,64,64,64" \
  --keys "math,gsm8k,metamath,aime,amc,olympiad,algebra,arc,openbook,geometry,number,theorem,logic,iq,reasoning,science,physics,chemistry,biology,probability,combinatorics"

scripts/k3d_env.sh run python -m knowledge3d.tools.phase25.long_run \
  --epochs 50 --limit 300 --eval-every 5 \
  --dims "64,64,64,64" \
  --keys "math,gsm8k,metamath,aime,amc,olympiad,algebra,arc,openbook,geometry,number,theorem,logic,iq,reasoning,science,physics,chemistry,biology,probability,combinatorics"

echo "Long run chain complete."

