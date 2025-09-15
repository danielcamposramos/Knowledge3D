from __future__ import annotations

from dataclasses import dataclass
from typing import List, Iterable
from pathlib import Path

from .sample_loader import SampleLoader  # type: ignore
from .meaning_training_pipeline import MeaningCentricTrainingPipeline  # type: ignore


_SHAPE_COMPLEXITY = {
    'tetrahedron': 1.0,
    'cube': 1.2,
    'octahedron': 1.5,
    'triangular_prism': 2.0,
    'pentagonal_prism': 2.2,
    'dodecahedron': 2.5,
    'icosahedron': 3.0,
    'truncated_icosahedron': 3.2,
    'snub_dodecahedron': 3.3,
    'great_rhombicuboctahedron': 3.4,
    'omnitruncated_icosahedron': 3.6,
    'hypersphere_projection': 4.0,
    'fractal_tree': 4.2,
}


@dataclass
class DevConfig:
    samples_dir: str = 'viewer/public/samples'
    models_dir: str = 'viewer/public/models'
    epochs: int = 50


class DevelopmentalTrainer:
    def __init__(self, cfg: DevConfig | None = None):
        self.cfg = cfg or DevConfig()
        self.loader = SampleLoader(self.cfg.samples_dir)

    def _all(self) -> List[dict]:
        # Recursively load all GLBs under samples_dir (including stage folders)
        base = Path(self.cfg.samples_dir)
        out: List[dict] = []
        for fp in base.rglob('*.glb'):
            try:
                s = self.loader.load_sample(fp)
                if s:
                    out.append({
                        'id': s.id,
                        'filepath': s.filepath,
                        'geometry': s.geometry,
                        'embedding': s.embedding,
                        'media_types': s.media_types,
                        'shape_type': s.shape_type,
                    })
            except Exception:
                continue
        return out

    def _complexity(self, shape: str) -> float:
        return _SHAPE_COMPLEXITY.get((shape or '').lower(), 2.0)

    def filter_samples_by_complexity(self, max_complexity: float) -> List[dict]:
        xs = []
        for s in self._all():
            c = self._complexity(str(s.get('shape_type') or ''))
            if c <= max(0.0, float(max_complexity)):
                xs.append(s)
        return xs

    def filter_samples_by_media_count(self, min_media: int = 1, max_media: int | None = None) -> List[dict]:
        xs = []
        for s in self._all():
            mt = s.get('media_types') or []
            n = len(mt)
            if n < int(min_media):
                continue
            if max_media is not None and n > int(max_media):
                continue
            xs.append(s)
        return xs

    def filter_samples_by_shape(self, shapes: Iterable[str]) -> List[dict]:
        wanted = {str(x).lower() for x in shapes}
        return [s for s in self._all() if str(s.get('shape_type') or '').lower() in wanted]

    def _train_on(self, samples: List[dict], out_name: str, **knobs):
        if not samples:
            print(f"⚠️ No samples for {out_name}; skipping.")
            return
        pipe = MeaningCentricTrainingPipeline(self.loader, **knobs)
        out = str(Path(self.cfg.models_dir) / out_name)
        pipe.train_on_samples(samples, epochs=int(self.cfg.epochs), out_path=out)

    # Stages
    def train_stage_1(self):
        # Sensorimotor: simple shapes + single modality (<=1)
        base = self.filter_samples_by_complexity(max_complexity=2.0)
        stage = [s for s in base if len(s.get('media_types') or []) <= 1 and str(s.get('shape_type') or '').lower() in {'tetrahedron','cube','octahedron'}]
        self._train_on(stage, out_name='stage1_sensorimotor.pth', honesty_threshold=0.9)
        print('✅ Stage 1 (Sensorimotor) completed')

    def train_stage_2(self):
        # Preoperational: hybrid shapes + 2 modalities
        stage = self.filter_samples_by_media_count(min_media=2, max_media=2)
        stage = [s for s in stage if str(s.get('shape_type') or '').lower() in {'triangular_prism','pentagonal_prism'}]
        self._train_on(stage, out_name='stage2_preoperational.pth', ray_threshold=0.7)
        print('✅ Stage 2 (Preoperational) completed')

    def train_stage_3(self):
        # Concrete Operational: complex shapes + 3+ modalities
        stage = self.filter_samples_by_media_count(min_media=3)
        stage = [s for s in stage if str(s.get('shape_type') or '').lower() in {'truncated_icosahedron','snub_dodecahedron','icosahedron'}]
        self._train_on(stage, out_name='stage3_concrete_operational.pth', use_rlwhf=True)
        print('✅ Stage 3 (Concrete Operational) completed')

    def train_stage_4(self):
        # Formal Operational: abstract + cross-modal fusion
        stage = self.filter_samples_by_shape(['hypersphere_projection','fractal_tree'])
        self._train_on(stage, out_name='stage4_formal_operational.pth', use_self_reflection=True)
        print('✅ Stage 4 (Formal Operational) completed')


def main():  # pragma: no cover
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', choices=['1','2','3','4','all'], default='1')
    ap.add_argument('--epochs', type=int, default=50)
    args = ap.parse_args()
    cfg = DevConfig(epochs=int(args.epochs))
    dt = DevelopmentalTrainer(cfg)
    if args.stage in ('1','all'): dt.train_stage_1()
    if args.stage in ('2','all'): dt.train_stage_2()
    if args.stage in ('3','all'): dt.train_stage_3()
    if args.stage in ('4','all'): dt.train_stage_4()


if __name__ == '__main__':  # pragma: no cover
    main()
