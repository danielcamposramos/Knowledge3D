# K3D automatic environment for interactive shells
if [ -f /home/daniel/miniforge/etc/profile.d/conda.sh ]; then
  . "/home/daniel/miniforge/etc/profile.d/conda.sh"
  conda activate k3d-cranium >/dev/null 2>&1 || true
fi
export PYTHONPATH=/K3D/Knowledge3D:${PYTHONPATH}
# Fused head eval defaults
export K3D_EVAL_MINIMAL=${K3D_EVAL_MINIMAL:-1}
export K3D_DISABLE_TEXT_MODALITY=${K3D_DISABLE_TEXT_MODALITY:-1}
export K3D_ENABLE_RPN_POLICY=${K3D_ENABLE_RPN_POLICY:-1}
