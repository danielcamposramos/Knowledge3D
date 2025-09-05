Memory‑Efficient RL (Unsloth‑style) — Early Integration

Goal
- Add a lightweight RL path to tune K3D’s intent policy with small memory.
- Seed rewards from session logs (e.g., goto resolution similarity, task success) and train a compact model.

References
- Unsloth Basics — Memory‑Efficient RL: https://docs.unsloth.ai/basics/memory-efficient-rl

Where it fits
- Logic Layer: RL policy complements supervised intent classifier.
- Data: ../Knowledge3D.local/logs/session-*.jsonl supply interactions and outcomes.
- Safety: Faith Engine threshold remains in effect for actions.

Usage
- Dump dataset or train a tiny RL‑tuned policy:
  - `python -m knowledge3d.rl.unsloth_adapter dump --logs ../Knowledge3D.local/logs --out ../Knowledge3D.local/models/intent_rl`
  - `python -m knowledge3d.rl.unsloth_adapter train --logs ../Knowledge3D.local/logs --out ../Knowledge3D.local/models/intent_rl --steps 1000`
- Load in live server: `/model load ../Knowledge3D.local/models/intent_rl` (HF format)

Notes
- This is a scaffold. For full Unsloth/TRL PPO, plug in the generated dataset; map rewards to shorter hops, higher sim, and safe outcomes.
- Keep training artifacts outside the repo; document steps in session reports.
