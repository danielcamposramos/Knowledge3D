from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List


class PostConsolidationReflector:
    def __init__(self, material_dir: str, critique_cycles: List[Dict[str, Any]]):
        self.material_dir = Path(material_dir)
        self.material_dir.mkdir(parents=True, exist_ok=True)
        self.critique_cycles: List[Dict[str, Any]] = list(critique_cycles or [])

    def reflect_on_consolidation(self) -> Dict[str, Any]:
        consolidated = [c for c in self.critique_cycles if str(c.get('status','')).startswith('consolidated')]
        discarded = [c for c in self.critique_cycles if str(c.get('status','')) == 'discarded']
        revised = [c for c in self.critique_cycles if int(c.get('revisions', 0)) > 0]
        reflection: Dict[str, Any] = {
            'cycle_id': int(datetime.now().timestamp()),
            'consolidated_count': len(consolidated),
            'discarded_count': len(discarded),
            'revised_count': len(revised),
            'avg_revisions': (sum(int(c.get('revisions', 0)) for c in revised) / len(revised)) if revised else 0.0,
            'avg_final_honesty': (sum(float(c.get('final_honesty', 0.0)) for c in consolidated) / len(consolidated)) if consolidated else 0.0,
            'patterns': self.identify_patterns(),
            'generated_at': datetime.now().isoformat(),
        }
        self.save_reflection_diary(reflection)
        return reflection

    def identify_patterns(self) -> List[str]:
        pats: List[str] = []
        lows = [c for c in self.critique_cycles if float(c.get('final_honesty', 0.0)) < 0.7]
        if lows:
            pats.append(f"Low honesty (<0.7) in {len(lows)} shapes — check embedding‑geometry alignment")
        highs = [c for c in self.critique_cycles if int(c.get('revisions', 0)) >= 2]
        if highs:
            pats.append(f"High revisions (≥2) in {len(highs)} shapes — consider pre‑generation honesty boost")
        return pats

    def generate_training_queries(self, max_q: int = 5) -> List[str]:
        qs: List[str] = []
        for c in self.critique_cycles:
            shape_name = Path(str(c.get('shape',''))).stem
            st = str(c.get('status',''))
            rev = int(c.get('revisions', 0))
            fh = float(c.get('final_honesty', 0.0))
            if st == 'discarded':
                qs.append(f"Why was shape {shape_name} discarded? What geometric property failed?")
            if rev > 0:
                qs.append(f"What honesty constraint was violated in revision {rev} of {shape_name}?")
            if fh < 0.8:
                qs.append(f"How to increase honesty for {shape_name} when embedding entropy is high?")
        # Deduplicate and cap
        out = []
        seen = set()
        for q in qs:
            if q not in seen:
                seen.add(q)
                out.append(q)
            if len(out) >= max_q:
                break
        return out

    def save_reflection_diary(self, reflection: Dict[str, Any]) -> str:
        diary_id = f"reflection_diary_cycle_{int(reflection.get('cycle_id', 0))}"
        p = self.material_dir / f"{diary_id}.json"
        data = {
            'type': 'reflection_diary',
            'title': f"Post-Consolidation Reflection — Cycle {reflection.get('cycle_id')}",
            'author': 'AI Self',
            'created_at': str(reflection.get('generated_at')),
            'content': reflection,
            'zone_placement': 'Zone 7 (Mirror Room)',
            'source': 'post_consolidation_reflection',
        }
        p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
        print(f"🧠 Saved Reflection Diary: {p}")
        return str(p)

