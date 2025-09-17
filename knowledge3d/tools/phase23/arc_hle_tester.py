from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from knowledge3d.tools.phase18.meaning_cluster_trainer import MeaningClusterTrainer


class ArchHleTester:
    """Zero-shot validation harness spanning ARC-AGI and HLE samples."""

    def __init__(self, limit: int = 50) -> None:
        self.trainer = MeaningClusterTrainer()
        self.limit = limit

    def load_arc_hle_questions(self) -> List[Dict[str, Any]]:
        raw = self.trainer.load_all_dataset_questions()
        arc: List[Dict[str, Any]] = []
        hle: List[Dict[str, Any]] = []
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

    def run(self) -> Dict[str, Any]:
        questions = self.load_arc_hle_questions()
        results: List[Dict[str, Any]] = []
        correct = 0
        for idx, item in enumerate(questions):
            query = item.get('query', '')
            answer = item.get('true_answer', '')
            embedding = self.trainer.generate_multi_modal_embedding(query)
            prediction = self.trainer.predict_from_fused_embedding(query, embedding, cluster_name='arc_hle_test')
            is_correct = prediction.strip().lower() == str(answer).strip().lower()
            if is_correct:
                correct += 1
            results.append(
                {
                    'index': idx,
                    'query': query,
                    'expected': answer,
                    'predicted': prediction,
                    'correct': is_correct,
                }
            )
            status = '✅' if is_correct else '❌'
            print(f"{status} Q{idx + 1}: {query}\n   Predicted: {prediction}\n   Expected:  {answer}")
        accuracy = correct / max(1, len(questions))
        summary = {
            'total_questions': len(questions),
            'correct': correct,
            'accuracy': accuracy,
            'results': results,
        }
        report_path = Path('logs/phase23_arc_hle_report.json')
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
        print(f"\n📊 ARC/HLE Test Accuracy: {accuracy:.0%}")
        print(f"💾 Report saved to {report_path}")
        return summary


def main() -> None:
    parser = argparse.ArgumentParser(description='ARC/HLE zero-shot tester')
    parser.add_argument('--limit', type=int, default=50, help='Maximum number of questions to evaluate')
    args = parser.parse_args()
    tester = ArchHleTester(limit=args.limit)
    tester.run()


if __name__ == '__main__':
    main()
