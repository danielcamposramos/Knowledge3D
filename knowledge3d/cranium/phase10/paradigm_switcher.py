from __future__ import annotations

import json
from typing import Dict, Optional

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except Exception:  # pragma: no cover
    torch = None
    nn = object  # type: ignore
    F = None  # type: ignore

from ..phase9.shape_recognizer import ShapeRecognizer  # type: ignore
from .multi_modality_fusion import MultiModalityFusion  # type: ignore
from .teacher_evaluator import TeacherEvaluator  # type: ignore


if torch is None:
    class ParadigmSwitcher:  # type: ignore
        def __init__(self):
            self.teacher_evaluator = TeacherEvaluator()
            self.modes = {
                "rlwhf": self.train_rlwhf,
                "qna": self.train_qna,
                "standard_rl": self.train_standard_rl,
                "supervised": self.train_supervised,
                "baby": self.train_baby,
            }
        def train(self, mode: str, data: Dict):
            if mode not in self.modes:
                raise ValueError(f"Unknown mode: {mode}")
            return self.modes[mode](data)
        def train_rlwhf(self, data: Dict):
            ai_resp = data.get("ai_response") or data.get("query") or ""
            ev = self.teacher_evaluator.evaluate_response(str(ai_resp))
            return ev
        def train_qna(self, data: Dict):
            return {"loss": 0.0, "note": "Torch unavailable"}
        def train_standard_rl(self, data: Dict):
            return {"loss": 0.0, "note": "Torch unavailable"}
        def train_supervised(self, data: Dict):
            return {"loss": 0.0, "note": "Torch unavailable"}
        def train_baby(self, data: Dict):
            return {"stage": data.get("stage", 1), "note": "Torch unavailable"}
else:
    class StudentModel(nn.Module):  # small head over fused modalities
        def __init__(self, input_dim: int = 512, num_classes: int = 2):
            super().__init__()
            self.fusion = MultiModalityFusion(input_dim=input_dim)
            self.fc1 = nn.Linear(input_dim, 256)
            self.fc2 = nn.Linear(256, num_classes)
            # RLWHF: scalar honesty bias parameter used for simple reward shaping
            self.honesty_bias = nn.Parameter(torch.zeros(1))
            self.relu = nn.ReLU()
        def forward(self, inputs: Dict[str, torch.Tensor]):
            z = self.fusion(inputs)
            x = self.relu(self.fc1(z))
            return self.fc2(x)

    class LiveTrainer:
        def __init__(self, student_model: StudentModel, teacher_evaluator: TeacherEvaluator):
            self.student_model = student_model
            self.teacher_evaluator = teacher_evaluator
        def train_on_live_query(self, query: str, true_answer: Optional[str] = None) -> Dict:
            # For now, echo-based student response; in production, hook LM head
            student_response = f"Echo: {query.strip()}"
            ev = self.teacher_evaluator.evaluate_response(student_response)
            return {"student_response": student_response, "evaluation": ev}

    class ParadigmSwitcher:
        def __init__(self, input_dim: int = 512, num_classes: int = 2, lr: float = 1e-3):
            self.student_model = StudentModel(input_dim=input_dim, num_classes=num_classes)
            self.teacher_evaluator = TeacherEvaluator()
            self.optimizer = torch.optim.Adam(self.student_model.parameters(), lr=lr)
            self.modes = {
                "rlwhf": self.train_rlwhf,
                "qna": self.train_qna,
                "standard_rl": self.train_standard_rl,
                "supervised": self.train_supervised,
                "baby": self.train_baby,
            }

        def train(self, mode: str, data: Dict):
            if mode not in self.modes:
                raise ValueError(f"Unknown mode: {mode}")
            return self.modes[mode](data)

        def train_rlwhf(self, data: Dict):
            # Evaluate provided AI response or synthesize from query
            ai_resp = data.get("ai_response")
            if not ai_resp:
                q = str(data.get("query", ""))
                ai_resp = f"Echo: {q}" if q else ""
            ev = self.teacher_evaluator.evaluate_response(str(ai_resp))
            # Reward shaping: map scores {-1,0.5,1} -> {0,0.5,1}
            raw = float(ev.get("score", -1.0))
            if raw <= -0.5:
                reward = 0.0
            elif raw < 1.0:
                reward = 0.5
            else:
                reward = 1.0
            # Simple loss: encourage honesty_bias -> 0 when reward high, -> 1 when reward low
            loss = (1.0 - reward) * (self.student_model.honesty_bias.sigmoid()).mean()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            return {"reward": reward, "loss": float(loss.item()), "evaluation": ev}

        def _ensure_tensor(self, x):
            if isinstance(x, torch.Tensor):
                return x.float()
            if isinstance(x, (list, tuple)):
                return torch.tensor(x, dtype=torch.float32)
            return torch.tensor([x], dtype=torch.float32)

        def train_qna(self, data: Dict):
            # classification over fused inputs
            inputs = data.get("inputs") or {"text": self._ensure_tensor(data.get("input", []))}
            logits = self.student_model(inputs)
            label = int(data.get("label", 0))
            loss = F.cross_entropy(logits, torch.tensor([label], dtype=torch.long))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            return {"loss": float(loss.item())}

        def train_standard_rl(self, data: Dict):
            inputs = data.get("inputs") or {"text": self._ensure_tensor(data.get("input", []))}
            logits = self.student_model(inputs)
            label = int(data.get("label", 0))
            probs = F.softmax(logits, dim=1)
            reward = 1.0 if int(torch.argmax(probs, dim=1).item()) == label else -1.0
            # Policy gradient surrogate (very simplified)
            logp = torch.log(probs[0, label] + 1e-9)
            loss = -reward * logp
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            return {"reward": reward, "loss": float(loss.item())}

        def train_supervised(self, data: Dict):
            inputs = data.get("inputs") or {"text": self._ensure_tensor(data.get("input", []))}
            logits = self.student_model(inputs)
            label = int(data.get("label", 0))
            loss = F.cross_entropy(logits, torch.tensor([label], dtype=torch.long))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            return {"loss": float(loss.item())}

        def train_baby(self, data: Dict):
            stage = int(data.get("stage", 1))
            if stage == 1:
                return self.train_supervised(data)
            if stage == 2:
                return self.train_rlwhf(data)
            if stage == 3:
                return self.train_standard_rl(data)
            return self.train_qna(data)

