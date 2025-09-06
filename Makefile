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

.PHONY: compile-mr clean-mr

# Generate Machine-Runtime (MR) sources outside the repo using codeopt
compile-mr:
	@echo "Compiling HR -> MR into ../Knowledge3D.local/mr ..."
	$(PY) -m codeopt --in k3dgen knowledge3d viewer --out ../Knowledge3D.local/mr --lang auto --stats
	@echo "Done."

clean-mr:
	rm -rf ../Knowledge3D.local/mr

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
