PY=python3

.PHONY: train-fast train-full scoreboard eval-logs

train-fast:
	$(PY) -m knowledge3d.tools.train_all --fast

train-full:
	$(PY) -m knowledge3d.tools.train_all

scoreboard:
	$(PY) -m knowledge3d.tools.train_session --gltf viewer/public/ai_books_basic.4k.umap.doors.glb --pairs 256 --door 96

eval-logs:
	$(PY) -m knowledge3d.models.eval_logs --logs ../Knowledge3D.local/logs

