from __future__ import annotations

from typing import Dict

try:
    import torch  # noqa: F401
except Exception:  # pragma: no cover
    torch = None

from .paradigm_switcher import ParadigmSwitcher  # type: ignore


class BabyCurriculum:
    def __init__(self):
        self.paradigm_switcher = ParadigmSwitcher()
        self.stage = 1

    def introduce_modality(self, modality: str):
        if self.stage == 1:
            allowed = {"text", "image", "audio"}
        elif self.stage == 2:
            allowed = {"text_image", "text_audio"}
        elif self.stage == 3:
            allowed = {"text_image_audio", "text_image_spatial"}
        else:
            allowed = {"all_modalities", "ontology"}
        if modality not in allowed:
            print(f"⚠️  Too early for {modality} — stay in Stage {self.stage}")
            return None
        return self.train_modality(modality)

    def train_modality(self, modality: str):
        data = self.get_data(modality)
        if self.stage == 1:
            return self.paradigm_switcher.train("supervised", data)
        elif self.stage == 2:
            return self.paradigm_switcher.train("rlwhf", data)
        elif self.stage == 3:
            return self.paradigm_switcher.train("standard_rl", data)
        else:
            return self.paradigm_switcher.train("qna", data)

    def advance_stage(self):
        if self.stage < 4:
            self.stage += 1
            print(f"🎉 Advanced to Stage {self.stage}")
        else:
            print("🎓 Fully grown — all modalities mastered")

    def get_data(self, modality: str) -> Dict:
        # Minimal synthetic data for demos
        if modality == "text":
            return {"input": [0.1, 0.9, 0.2, 0.8], "label": 1}
        if modality == "image":
            return {"input": [0.9, 0.1, 0.2, 0.0], "label": 0}
        if modality == "audio":
            return {"input": [0.3, 0.4, 0.5, 0.6], "label": 1}
        if modality == "text_image":
            return {"query": "Describe the picture.", "ai_response": "The sky is blue."}
        if modality == "text_audio":
            return {"query": "What sound is this?", "ai_response": "It is a bell."}
        if modality in {"text_image_audio", "text_image_spatial"}:
            return {"input": [0.2, 0.2, 0.7, 0.9], "label": 1}
        if modality in {"all_modalities", "ontology"}:
            return {"input": [0.5, 0.4, 0.3, 0.2], "label": 0}
        return {"input": [0.0, 0.0, 0.0, 0.0], "label": 0}


def main():  # pragma: no cover
    import argparse, json
    ap = argparse.ArgumentParser(description="Baby curriculum stage manager")
    ap.add_argument("--check_correct", type=int, default=None, help="Number of queries expected correct")
    ap.add_argument("--advance_if_ready", action="store_true")
    ap.add_argument("--stage", type=int, default=None)
    args = ap.parse_args()
    if args.check_correct is not None:
        # Read last session log
        from pathlib import Path
        p = Path("logs/phase10.6_last_session.json")
        if not p.exists():
            print("⚠️ No live session results found")
            return
        data = json.loads(p.read_text(encoding="utf-8"))
        correct = int(data.get("correct_count", 0))
        num = int(data.get("num_queries", 0))
        all_correct = bool(data.get("all_correct", False))
        cur_stage = int(data.get("stage", 1))
        if all_correct and correct >= int(args.check_correct):
            if cur_stage >= 4:
                print("🎓 Fully grown — all modalities mastered")
            else:
                print(f"🎉 Advanced to Stage {cur_stage+1}")
        else:
            print(f"⚠️ Stay in Stage {cur_stage}")
        return
    # Default: demonstrate a single modality at stage
    bc = BabyCurriculum()
    if args.stage:
        bc.stage = int(args.stage)
    res = bc.introduce_modality("text")
    if res is not None:
        print(res)


if __name__ == "__main__":  # pragma: no cover
    main()
