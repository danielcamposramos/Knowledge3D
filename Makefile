PY?=python3

.PHONY: train-fast train-full scoreboard eval-logs

train-fast:
	$(PY) -m knowledge3d.tools.train_all --fast

train-full:
	$(PY) -m knowledge3d.tools.train_all

scoreboard:
	$(PY) -m knowledge3d.tools.train_session --gltf viewer/public/ai_books_basic.4k.umap.doors.glb --pairs 256 --door 96

eval-logs:
	$(PY) -m knowledge3d.models.eval_logs --logs ../Knowledge3D.local/logs

.PHONY: compile-mr compile-mr-core compile-mr-trainers compile-mr-all clean-mr mr-report

# === Dual-Code (HR/MR) Targets ===
# See docs/DUAL_CODE_STRATEGY.md for when/why to use each tier

# Tier 1: Hot-path modules (fused head, PTX ops, skills)
compile-mr-core:
	@echo "Compiling Tier 1 (core hot-path) HR -> MR ..."
	$(PY) -m codeopt \
	  --in knowledge3d/cranium/fused_head.py \
	       knowledge3d/cranium/ptx \
	       knowledge3d/skills \
	       knowledge3d/bridge/live_server.py \
	  --out ../Knowledge3D.local/mr --lang py --stats
	@echo "Done. Use PYTHONPATH=../Knowledge3D.local/mr:. to prefer MR modules."

# Tier 2: Training scripts (for multi-instance parallel runs)
compile-mr-trainers:
	@echo "Compiling Tier 2 (trainers) HR -> MR ..."
	$(PY) -m codeopt \
	  --in knowledge3d/tools/phase18 \
	       knowledge3d/tools/phase25 \
	       knowledge3d/tools/*_evaluator.py \
	       knowledge3d/tools/*_trainer.py \
	  --out ../Knowledge3D.local/mr --lang py --stats
	@echo "Done."

# Tier 3: Full repository (legacy behavior; use for production edge deployment)
compile-mr-all:
	@echo "Compiling HR -> MR into ../Knowledge3D.local/mr (full repo) ..."
	$(PY) -m codeopt --in k3dgen knowledge3d viewer --out ../Knowledge3D.local/mr --lang auto --stats
	@echo "Done."

# Alias for backward compatibility
compile-mr: compile-mr-all

# Clean all MR outputs
clean-mr:
	rm -rf ../Knowledge3D.local/mr

# Report: show memory savings by tier
mr-report:
	@echo "=== MR Savings Report ==="
	@if [ -d ../Knowledge3D.local/mr ]; then \
		echo "Tier 1 (core):"; \
		du -sh ../Knowledge3D.local/mr/knowledge3d/cranium/fused_head.py 2>/dev/null || echo "  Not compiled"; \
		du -sh ../Knowledge3D.local/mr/knowledge3d/cranium/ptx 2>/dev/null || echo "  Not compiled"; \
		echo "Tier 2 (trainers):"; \
		du -sh ../Knowledge3D.local/mr/knowledge3d/tools/phase25 2>/dev/null || echo "  Not compiled"; \
		echo "Total MR size:"; \
		du -sh ../Knowledge3D.local/mr; \
	else \
		echo "MR directory not found. Run 'make compile-mr-core' first."; \
	fi

.PHONY: session-80k-long build-120k session-120k-medium

# Long 80k session (writes report under docs/reports/training)
session-80k-long:
	$(PY) -m knowledge3d.tools.train_session \
	  --gltf ../Knowledge3D.local/datasets/ai_compendium.80k.pca.doors.glb \
	  --pairs 2048 --door 1024 --out-dir docs/reports/training

# Build 120k compendium (local-only artifacts under ../Knowledge3D.local/datasets)

build-120k:
	$(PY) -m knowledge3d.tools.build_corpus --target 120000 --out data/ai_compendium_120k.txt
	$(PY) -m knowledge3d.tools.text_to_vectors \
	  --text data/ai_compendium_120k.txt \
	  --out ../Knowledge3D.local/datasets/ai_compendium_120k_vectors.csv \
	  --dims 512
	$(PY) -m k3dgen ../Knowledge3D.local/datasets/ai_compendium_120k_vectors.csv \
	  --gltf ../Knowledge3D.local/datasets/ai_compendium.120k.pca.glb --k 5 --reducer pca --emb-precision f16
	$(PY) -m knowledge3d.tools.mark_doors \
	  --input ../Knowledge3D.local/datasets/ai_compendium.120k.pca.glb \
	  --output ../Knowledge3D.local/datasets/ai_compendium.120k.pca.doors.glb \
	  --doors 1920 --trail true

# Medium 120k session
session-120k-medium:
	$(PY) -m knowledge3d.tools.train_session \
	  --gltf ../Knowledge3D.local/datasets/ai_compendium.120k.pca.doors.glb \
	  --pairs 512 --door 256 --out-dir docs/reports/training
