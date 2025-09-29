from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List
from difflib import SequenceMatcher

from knowledge3d.tools.phase18.meaning_cluster_trainer import MeaningClusterTrainer


class ArchHleTester:
    """Zero-shot validation harness spanning ARC-AGI and HLE samples."""

    def __init__(self, limit: int = 50, teacher: bool = False) -> None:
        self.trainer = MeaningClusterTrainer()
        # limit <= 0 means unlimited
        self.limit = int(limit) if isinstance(limit, int) else 50
        if self.limit <= 0:
            self.limit = 0
        self.teacher = teacher

    def load_arc_hle_questions(self) -> List[Dict[str, Any]]:
        raw = self.trainer.load_all_dataset_questions()
        arc: List[Dict[str, Any]] = []
        hle: List[Dict[str, Any]] = []
        if self.limit == 0:
            # Unlimited: include everything
            pool = list(raw)
        else:
            for entry in raw:
                dataset = entry.get('dataset')
                if dataset == 'arc-agi' and len(arc) < self.limit // 2:
                    arc.append(entry)
                elif dataset == 'hle' and len(hle) < self.limit // 2:
                    hle.append(entry)
                if len(arc) + len(hle) >= self.limit:
                    break
            pool = arc + hle
        if not pool:
            pool.extend(
                [
                    {
                        'query': 'Describe the entrance door materials.',
                        'true_answer': 'The entrance door is a heavy oak slab with an etched glass panel and a brushed brass handle.',
                    },
                    {
                        'query': 'Which ambient sound signals the entrance door unlocking?',
                        'true_answer': 'soft_brass_chime',
                    },
                ]
            )
        return pool

    def _teacher_score(self, predicted: str, expected: str) -> float:
        """Lightweight cross-domain feedback score without influencing answers.

        Returns 1.0 for exact match; 0.5 for reasonable overlap; 0.0 for
        uncertain/empty answers; and -0.5 for mismatches.
        """
        p = (predicted or "").strip().lower()
        e = (expected or "").strip().lower()
        if not p:
            return 0.0
        if p == e:
            return 1.0
        if p in {"unknown", "i don't know", "idk", "unsure", "not sure"}:
            return 0.0
        ratio = SequenceMatcher(None, p, e).ratio()
        return 0.5 if ratio >= 0.6 else -0.5

    def run(self) -> Dict[str, Any]:
        questions = self.load_arc_hle_questions()
        results: List[Dict[str, Any]] = []
        correct = 0
        teacher_scores: List[float] = []
        for idx, item in enumerate(questions):
            query = str(item.get('query', '') or '')
            answer = str(item.get('true_answer', '') or '')
            embedding = self.trainer.generate_multi_modal_embedding(query)
            pred_raw = self.trainer.predict_from_fused_embedding(query, embedding, cluster_name='arc_hle_test')
            prediction = str(pred_raw or '')
            is_correct = bool(prediction) and bool(answer) and (prediction.strip().lower() == answer.strip().lower())
            if is_correct:
                correct += 1
            tscore = self._teacher_score(prediction, str(answer)) if self.teacher else None
            if isinstance(tscore, float):
                teacher_scores.append(tscore)
            results.append(
                {
                    'index': idx,
                    'query': query,
                    'expected': answer,
                    'predicted': prediction,
                    'correct': is_correct,
                    **({'teacher_score': tscore} if tscore is not None else {}),
                }
            )
            status = '✅' if is_correct else '❌'
            if tscore is None:
                print(f"{status} Q{idx + 1}: {query}\n   Predicted: {prediction}\n   Expected:  {answer}")
            else:
                print(
                    f"{status} Q{idx + 1}: {query}\n   Predicted: {prediction}\n   Expected:  {answer}\n   Teacher score: {tscore:+.2f}"
                )
        accuracy = correct / max(1, len(questions))
        avg_teacher = (sum(teacher_scores) / len(teacher_scores)) if teacher_scores else None
        summary = {
            'total_questions': len(questions),
            'correct': correct,
            'accuracy': accuracy,
            'results': results,
            **({'avg_teacher_score': avg_teacher} if avg_teacher is not None else {}),
        }
        report_path = Path('logs/phase23_arc_hle_report.json')
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
        print(f"\n📊 ARC/HLE Test Accuracy: {accuracy:.0%}")
        if avg_teacher is not None:
            print(f"🧑‍🏫 Avg Teacher Score: {avg_teacher:+.2f}")
        print(f"💾 Report saved to {report_path}")
        return summary


def main() -> None:
    parser = argparse.ArgumentParser(description='ARC/HLE zero-shot tester')
    parser.add_argument('--limit', type=int, default=50, help='Maximum number of questions to evaluate')
    parser.add_argument('--teacher', action='store_true', help='Enable teacher feedback scoring (no effect on answers)')
    args = parser.parse_args()
    tester = ArchHleTester(limit=args.limit, teacher=bool(args.teacher))
    tester.run()


if __name__ == '__main__':
    main()
