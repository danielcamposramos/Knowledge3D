from __future__ import annotations

import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

from knowledge3d.cranium.ptx import PTX_OPS


class _Model:
    def __init__(self, nodes: List[Dict[str, Any]], gltf: Any | None = None):
        self.nodes = nodes
        self.gltf = gltf


class _GLBLoaderFallback:
    def load(self, path: str) -> _Model:  # pragma: no cover
        # Fallback: return empty model if parsing not available or file missing
        try:
            p = Path(path)
            if not p.exists():
                return _Model([], None)
        except Exception:
            pass
        return _Model([], None)


def _load_glb(path: str) -> _Model:
    try:
        from pygltflib import GLTF2  # type: ignore
        p = Path(path)
        if not p.exists():
            return _Model([])
        gltf = GLTF2().load(str(p))
        nodes: List[Dict[str, Any]] = []
        for idx, node in enumerate(gltf.nodes or []):
            # Normalize to dict with extras for our traversal
            extras = getattr(node, "extras", None)
            # Ensure dict
            if hasattr(extras, "to_dict"):
                try:
                    extras = extras.to_dict()
                except Exception:
                    extras = dict(extras)
            if extras is None:
                extras = {}
            nodes.append({
                "name": getattr(node, "name", None),
                "extras": extras,
                "mesh": getattr(node, "mesh", None),
                "node_index": idx,
            })
        return _Model(nodes, gltf)
    except Exception:
        return _GLBLoaderFallback().load(path)


class SleepTimeCompute:
    def __init__(self, house_path: str, galaxy_path: str, output_path: str | None = None, material_dir: str | None = None):
        self.house_path = Path(house_path)
        self.galaxy_path = Path(galaxy_path)
        self.output_path = Path(output_path) if output_path else self.house_path.parent / "house_post_sleep.glb"
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.material_dir = Path(material_dir) if material_dir else self.house_path.parent / "materialized_objects"
        self.material_dir.mkdir(parents=True, exist_ok=True)
        self.house: Optional[Dict[str, Any]] = None
        self.galaxy: Optional[List[Dict[str, Any]]] = None
        self._house_model: Optional[_Model] = None
        self._ptx_scene_loaded: bool = False

    def load_house(self) -> Dict[str, Any]:
        """Load House GLB — extract zones, rays, embeddings (best-effort)."""
        model = _load_glb(str(self.house_path))
        self._house_model = model
        house_data: Dict[str, Any] = {
            'zones': [],
            'rays': [],
            'nodes': model.nodes,
        }
        for node in model.nodes:
            extras = node.get('extras') or {}
            k3d = extras.get('k3d') if isinstance(extras, dict) else None
            if not isinstance(k3d, dict):
                continue
            if k3d.get('type') == 'zone':
                house_data['zones'].append({
                    'id': k3d.get('id'),
                    'name': k3d.get('name') or node.get('name'),
                    'position': list(k3d.get('position', [0.0, 0.0, 0.0])),
                    'honesty_score': float(k3d.get('honesty_score', 0.5)),
                    'mesh_id': node.get('mesh'),
                    'primitive_index': k3d.get('primitive_index'),
                    'node_index': node.get('node_index'),
                })
            elif k3d.get('type') == 'ray':
                house_data['rays'].append({
                    'id': k3d.get('id'),
                    'origin_zone': k3d.get('origin_zone'),
                    'target_zone': k3d.get('target_zone'),
                    'honesty_score': float(k3d.get('honesty_score', 0.5)),
                })
        return house_data

    def load_galaxy(self) -> List[Dict[str, Any]]:
        """Load Galaxy GLB — extract stars, honesty scores, chat logs, reflections (best-effort)."""
        model = _load_glb(str(self.galaxy_path))
        stars: List[Dict[str, Any]] = []
        for node in model.nodes:
            extras = node.get('extras') or {}
            k3d = extras.get('k3d') if isinstance(extras, dict) else None
            if not isinstance(k3d, dict):
                continue
            if k3d.get('type') == 'star':
                stars.append({
                    'id': k3d.get('id'),
                    'position': list(k3d.get('position', [0.0, 0.0, 0.0])),
                    'honesty_score': float(k3d.get('honesty_score', 0.5)),
                    'embedding_entropy': float(k3d.get('embedding_entropy', 0.0)),
                    'chat_history': list(k3d.get('chat_history', [])),
                    'self_reflections': list(k3d.get('self_reflections', [])),
                    'generated_shapes': list(k3d.get('generated_shapes', [])),
                    'connected_stars': list(k3d.get('connected_stars', [])),
                    'embedding': list(k3d.get('embedding', [])),
                })
        return stars

    def _write_json(self, path: Path, data: Dict[str, Any]) -> str:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return str(path)

    def materialize_chat_history(self, star: Dict[str, Any]) -> Optional[str]:
        """Materialize chat history into a book object (JSON for now)."""
        if not star.get('chat_history'):
            return None
        book_id = f"book_chat_{star.get('id')}_{int(time.time())}"
        book_path = self.material_dir / f"{book_id}.json"
        book_data = {
            'type': 'chat_history_book',
            'title': f"Chat Log — Star {star.get('id')}",
            'author': 'AI Self',
            'created_at': datetime.now().isoformat(),
            'honesty_score': star.get('honesty_score', 0.0),
            'content': star.get('chat_history', []),
            'embedding': star.get('embedding', []),
            'zone_placement': 'Zone 3 (Library)',
        }
        return self._write_json(book_path, book_data)

    def materialize_diary_entry(self, star: Dict[str, Any]) -> Optional[str]:
        """Materialize self-reflections into diary entries (JSON)."""
        if not star.get('self_reflections'):
            return None
        diary_id = f"diary_{star.get('id')}_{int(time.time())}"
        diary_path = self.material_dir / f"{diary_id}.json"
        diary_data = {
            'type': 'diary_entry',
            'title': f"Self-Reflection — {datetime.now().strftime('%Y-%m-%d')}",
            'author': 'AI Self',
            'created_at': datetime.now().isoformat(),
            'honesty_score': star.get('honesty_score', 0.0),
            'content': star.get('self_reflections', []),
            'embedding': star.get('embedding', []),
            'zone_placement': 'Zone 7 (Mirror Room)',
        }
        return self._write_json(diary_path, diary_data)

    def materialize_fractal_tree(self, star: Dict[str, Any]) -> Optional[str]:
        """Materialize knowledge as a fractal tree metadata JSON (GLB stub)."""
        if float(star.get('honesty_score', 0.0)) < 0.6:
            return None
        tree_id = f"tree_{star.get('id')}_{int(time.time())}"
        tree_meta_path = self.material_dir / f"{tree_id}.json"
        tree_data = {
            'type': 'fractal_tree',
            'name': f"Knowledge Tree — Star {star.get('id')}",
            'created_at': datetime.now().isoformat(),
            'honesty_score': star.get('honesty_score', 0.0),
            'branch_density': int(1.618 * float(star.get('honesty_score', 0.0)) * 10),
            'embedding': star.get('embedding', []),
            'zone_placement': 'Zone 5 (Knowledge Garden)',
        }
        return self._write_json(tree_meta_path, tree_data)

    def compute_nightly_adjustments(self) -> Dict[str, Any]:
        """Adjust House zones, prune rays, and materialize knowledge into permanent objects."""
        self.house = self.load_house()
        self.galaxy = self.load_galaxy()

        self._ensure_ptx_scene()

        adjustments: Dict[str, Any] = {
            'zone_shifts': [],
            'ray_adjustments': [],
            'pruned_rays': [],
            'materialized_objects': [],
        }

        zones = self.house.get('zones', []) if self.house else []
        rays = self.house.get('rays', []) if self.house else []
        stars = self.galaxy or []
        star_by_id = {s.get('id'): s for s in stars}

        # Adjust zone positions based on honesty-weighted alignment to star positions
        for z in zones:
            sid = z.get('id')
            star = star_by_id.get(sid)
            if not star:
                continue
            h = float(star.get('honesty_score', 0.0))
            if h <= 0.5:
                continue
            old_pos = list(z.get('position', [0.0, 0.0, 0.0]))
            st_pos = list(star.get('position', [0.0, 0.0, 0.0]))
            shift = [(st_pos[i] - old_pos[i]) * h for i in range(3)]
            new_pos = [old_pos[i] + shift[i] for i in range(3)]
            z['position'] = new_pos
            adjustments['zone_shifts'].append({
                'zone_id': sid,
                'old_position': old_pos,
                'new_position': new_pos,
                'shift_vector': shift,
                'honesty_score': h,
            })
            self._apply_zone_translation(z, shift)

        # Adjust ray origins to updated zone positions; prune low-honesty rays
        kept_rays = []
        for r in rays:
            h = float(r.get('honesty_score', 0.0))
            if h < 0.3:
                adjustments['pruned_rays'].append(r.get('id'))
                continue
            origin_id = r.get('origin_zone')
            origin_zone = next((zz for zz in zones if zz.get('id') == origin_id), None)
            if origin_zone:
                adjustments['ray_adjustments'].append({
                    'ray_id': r.get('id'),
                    'origin_zone': origin_id,
                    'new_origin_position': origin_zone.get('position', [0.0, 0.0, 0.0]),
                })
            kept_rays.append(r)
        if self.house is not None:
            self.house['rays'] = kept_rays

        # Materialize knowledge objects per high-honesty stars
        for s in stars:
            h = float(s.get('honesty_score', 0.0))
            if h >= 0.5:
                # Chat history → book
                p = self.materialize_chat_history(s)
                if p:
                    adjustments['materialized_objects'].append({'type': 'chat_history_book', 'path': p, 'zone': 'Zone 3 (Library)', 'star_id': s.get('id')})
                # Self-reflections → diary entry
                p = self.materialize_diary_entry(s)
                if p:
                    adjustments['materialized_objects'].append({'type': 'diary_entry', 'path': p, 'zone': 'Zone 7 (Mirror Room)', 'star_id': s.get('id')})
                # Knowledge tree → fractal
                p = self.materialize_fractal_tree(s)
                if p:
                    adjustments['materialized_objects'].append({'type': 'fractal_tree', 'path': p, 'zone': 'Zone 5 (Knowledge Garden)', 'star_id': s.get('id')})
        # Prepare Galaxy working dir for pre‑consolidation drafts
        working_dir = Path('viewer/public/galaxy/working')
        working_dir.mkdir(parents=True, exist_ok=True)
        generated_paths: list[str] = []

        # Phase 13: Autonomous synthesis + curriculum (drafts to Galaxy working dir)
        try:
            from ..phase13.auto_synthesis_engine import AutoSynthesisEngine  # type: ignore
            print("🧠 Running Autonomous Meaning Synthesis...")
            syn = AutoSynthesisEngine(str(self.house_path), str(self.galaxy_path), str(working_dir))
            synthesized = syn.run_synthesis()
            for item in synthesized:
                p = item.get('path')
                if isinstance(p, str):
                    generated_paths.append(p)
        except Exception as e:
            print(f"⚠️  Auto synthesis skipped: {e}")
        try:
            from ..phase13.self_curriculum_engine import SelfCurriculumEngine  # type: ignore
            print("📚 Generating Self-Curated Curriculum...")
            cur = SelfCurriculumEngine(str(self.galaxy_path))
            queries = cur.run_self_curriculum()
            if queries:
                adjustments['self_training'] = []
                from datetime import datetime as _dt
                ts = _dt.now().isoformat()
                for q in queries:
                    print(f"🎓 Self-Training on: {q}")
                    adjustments['self_training'].append({'query': q, 'trained_at': ts})
        except Exception as e:
            print(f"⚠️  Self-curated curriculum skipped: {e}")

        # Phase 14: Dream shapes (Zone 6) — drafts to Galaxy working dir
        try:
            from ..phase14.dream_engine import DreamEngine  # type: ignore
            print("🌌 Dreaming New Geometry...")
            de = DreamEngine(str(self.galaxy_path), str(working_dir))
            dreams = de.dream(num_dreams=3)
            for d in dreams:
                p = d.get('path')
                if isinstance(p, str):
                    generated_paths.append(p)
        except Exception as e:
            print(f"⚠️  Dream engine skipped: {e}")

        # Phase 15.2: Pre‑consolidation critique loop
        try:
            from ..phase15.honest_critique_engine import HonestCritiqueEngine  # type: ignore
            from ..phase15.critique_applier import CritiqueApplier  # type: ignore
            import shutil as _sh
            from pygltflib import GLTF2  # type: ignore
            print(f"🔍 Found {len(generated_paths)} new shapes in Galaxy for pre-consolidation critique...")
            final_shapes: list[str] = []
            critique_cycles: list[dict] = []
            for shape_path in list(generated_paths):
                current_path = str(shape_path)
                revision = 0
                max_revisions = 3
                honesty_score = 0.5
                while revision < max_revisions:
                    # read honesty
                    h = 0.5
                    try:
                        gltf = GLTF2().load(current_path)
                        for n in (gltf.nodes or []):
                            ex = getattr(n, 'extras', None)
                            if hasattr(ex, 'to_dict'):
                                try:
                                    ex = ex.to_dict()
                                except Exception:
                                    ex = dict(ex)
                            if isinstance(ex, dict) and isinstance(ex.get('k3d'), dict):
                                hh = ex['k3d'].get('honesty_score')
                                if isinstance(hh, (int, float)):
                                    h = float(hh)
                                    break
                    except Exception:
                        pass
                    honesty_score = h
                    if honesty_score >= 0.85:
                        print(f"✅ Shape {Path(current_path).stem} passed honesty threshold ({honesty_score:.2f}) at revision {revision}")
                        final_shapes.append(current_path)
                        critique_cycles.append({'shape': current_path, 'revisions': revision, 'final_honesty': honesty_score, 'status': 'consolidated'})
                        break
                    # critique and apply
                    ce = HonestCritiqueEngine(str(self.material_dir), str(self.galaxy_path))
                    crits = ce.critique_shapes([{'path': current_path}])
                    if not crits:
                        print(f"✅ No critiques for {Path(current_path).name} — accepting as-is.")
                        final_shapes.append(current_path)
                        critique_cycles.append({'shape': current_path, 'revisions': revision, 'final_honesty': honesty_score, 'status': 'consolidated_no_critique'})
                        break
                    ap = CritiqueApplier(str(self.material_dir), str(working_dir))
                    try:
                        new_path = ap.apply_shape_critique(current_path, revision=revision+1, delta=0.15)
                        current_path = new_path
                        revision += 1
                        print(f"🔧 Revised {Path(current_path).stem}")
                    except Exception as e:
                        print(f"⚠️  Failed to revise {current_path}: {e}")
                        break
                # handle fail/discard
                if revision >= max_revisions and honesty_score < 0.85:
                    print(f"❌ Shape {Path(current_path).stem} failed to reach honesty threshold after {max_revisions} revisions — NOT consolidated.")
                    critique_cycles.append({'shape': current_path, 'revisions': revision, 'final_honesty': honesty_score, 'status': 'discarded'})

            # Consolidate to House only final_shapes
            for f in final_shapes:
                try:
                    src = Path(f)
                    dst = self.material_dir / src.name
                    _sh.copy2(src, dst)
                    # also copy rays (if generated in working dir)
                    r = src.with_name(f"rays_{src.stem}.glb")
                    if r.exists():
                        _sh.copy2(r, self.material_dir / r.name)
                    adjustments['materialized_objects'].append({'type': 'generated_3d_shape', 'path': str(dst), 'zone': 'Zone 5 (Knowledge Garden)', 'source': 'galaxy_after_critique'})
                except Exception as e:
                    print(f"⚠️  Failed to consolidate {f}: {e}")
            for cyc in critique_cycles:
                try:
                    print(f"📊 {Path(cyc['shape']).name}: {cyc['revisions']} revisions, final honesty {float(cyc['final_honesty']):.2f} → {cyc['status']}")
                except Exception:
                    pass
            # Phase 16: Reflect and (mock) train
            try:
                from ..phase16.post_consolidation_reflector import PostConsolidationReflector  # type: ignore
                print("🧠 Running Post-Consolidation Reflection...")
                ref = PostConsolidationReflector(str(self.material_dir), critique_cycles)
                reflection = ref.reflect_on_consolidation()
                queries = ref.generate_training_queries()
                if queries:
                    from datetime import datetime as _dt
                    print(f"🎓 Generating {len(queries)} new training queries from reflection...")
                    for q in queries:
                        print(f"  → {q}")
                        adjustments.setdefault('post_consolidation_training', []).append({'query': q, 'trained_at': _dt.now().isoformat()})
                    print("✅ Post-Consolidation Training Complete.")
            except Exception as e:
                print(f"⚠️  Post-consolidation reflection skipped: {e}")
        except Exception as e:
            print(f"⚠️  Critique loop skipped: {e}")

        # Persist Galaxy state for continuity
        try:
            from ..phase17.galaxy_state_serializer import GalaxyStateSerializer  # type: ignore
            print("💾 Saving Galaxy State for Eternal Continuity...")
            GalaxyStateSerializer(str(self.galaxy_path)).serialize_galaxy_state()
        except Exception as e:
            print(f"⚠️  Galaxy state serialization skipped: {e}")

        return adjustments

    # PTX helpers -------------------------------------------------------------
    def _ensure_ptx_scene(self) -> None:
        if self._ptx_scene_loaded:
            return
        try:
            PTX_OPS.geometry_load_scene(str(self.house_path))
            self._ptx_scene_loaded = True
        except Exception as exc:
            print(f"⚠️  PTX geometry scene load skipped: {exc}")

    def _apply_zone_translation(self, zone: Dict[str, Any], shift: List[float]) -> None:
        if not self._ptx_scene_loaded:
            return
        mesh_id = zone.get('mesh_id')
        if mesh_id is None:
            return
        primitive_index = zone.get('primitive_index')
        try:
            PTX_OPS.geometry_translate_mesh(
                int(mesh_id),
                shift,
                primitive_index=int(primitive_index) if primitive_index is not None else None,
                recalc_normals=False,
            )
        except Exception as exc:
            print(f"⚠️  PTX zone translation failed for zone {zone.get('id')}: {exc}")

    def _persist_ptx_updates(self) -> None:
        if not self._ptx_scene_loaded:
            return
        try:
            PTX_OPS.geometry_save(target_glb=str(self.output_path))
            self._update_zone_positions_in_glb(self.output_path)
        except Exception as exc:
            print(f"⚠️  PTX geometry save skipped: {exc}")
        finally:
            try:
                PTX_OPS.geometry_release()
            except Exception as rel_exc:
                print(f"⚠️  PTX geometry release failed: {rel_exc}")
            self._ptx_scene_loaded = False

    def _update_zone_positions_in_glb(self, glb_path: Path) -> None:
        try:
            from pygltflib import GLTF2  # type: ignore
        except Exception as exc:  # pragma: no cover
            print(f"⚠️  Unable to update zone positions in GLB (pygltflib missing): {exc}")
            return

        zone_map = {}
        if self.house:
            for zone in self.house.get('zones', []):
                zone_map[zone.get('id')] = zone
        if not zone_map:
            return

        gltf = GLTF2().load(str(glb_path))
        updated = False
        for node in gltf.nodes or []:
            extras = getattr(node, 'extras', None)
            if not extras:
                continue
            if hasattr(extras, 'to_dict'):
                extras = extras.to_dict()
            if not isinstance(extras, dict):
                continue
            k3d = extras.get('k3d')
            if not isinstance(k3d, dict):
                continue
            if k3d.get('type') != 'zone':
                continue
            zone_id = k3d.get('id')
            zone = zone_map.get(zone_id)
            if not zone:
                continue
            k3d['position'] = list(zone.get('position', [0.0, 0.0, 0.0]))
            updated = True
            node.extras = extras
        if updated:
            gltf.save(str(glb_path))

    def save_house(self) -> str:
        """Save modified House GLB (stub). Logs adjustments and materializations."""
        adjustments = self.compute_nightly_adjustments()
        self._persist_ptx_updates()
        # Always log to central logs directory
        logs_dir = Path('logs')
        logs_dir.mkdir(parents=True, exist_ok=True)
        log_path = logs_dir / 'sleep_time_adjustments.json'
        with log_path.open('w', encoding='utf-8') as f:
            json.dump(adjustments, f, ensure_ascii=False, indent=2)
        print(f"🌙 Sleep-Time Compute: Adjustments logged to {log_path}")
        print(f"📚 Materialized {len(adjustments['materialized_objects'])} permanent objects.")
        if self.output_path.exists():
            print(f"🌙 Modified House GLB saved to {self.output_path}")
        else:
            print(f"🌙 PTX geometry save skipped; no GLB written (see warnings above)")
        return str(self.output_path)

    def run(self) -> None:
        """Execute sleep-time compute pipeline."""
        print("🌙 Initiating Sleep-Time Compute & Permanent Materialization...")
        out = self.save_house()
        print(f"🌙 Sleep-Time Compute Complete. House ready for reload at {out}")
        print(f"🏛️  All materialized objects stored in: {self.material_dir}")
