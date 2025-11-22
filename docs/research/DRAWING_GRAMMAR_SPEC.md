# Drawing Grammar Specification (Procedural Composition)

Purpose: mirror the language hierarchy (letters → words → phrases → text → book) in the visual domain using procedural primitives and PTX pipelines.

Layers:
1) **Primitives (atomic)**: line, arc, quad/cubic Bézier, circle/ellipse, rectangle, triangle. Procedural opcodes already in `rpn_executor` / `procedural_glyph_rasterizer`.
2) **Stroke Compositions (2D)**: grouping primitives into strokes with width/color/alpha; basic 2D transforms (translate/scale/rotate/skew), stroke joins/caps, gradients, patterns.
3) **Shapes/Icons (2D motifs)**: composed strokes forming reusable motifs (arrows, UI glyphs, icons), stored as procedural programs; Matryoshka embeddings regenerable from execution.
4) **Scenes/Layouts (2.5D)**: compositions of motifs with layering, clipping, simple depth ordering; LOD toggles; procedural effects (shadow/blur) if available in stack.
5) **3D Forms (basic depth)**: simple extrusions/lathe/revolve of 2D shapes; depth transforms; still procedural, backed by PTX geometry kernels.
6) **Narratives/Illustrations**: multi-object compositions (akin to “phrases”); references to shapes/icons/scenes with spatial/semantic relations.
7) **Books/Collections**: ordered sets of illustrations/scenes with metadata; analogous to text “book” layer.

Principles:
- Meaning-first nodes: same concept → one star with multiple style variants (e.g., “arrow_right” in thin/bold/fill); different meanings → separate stars even if visually similar.
- Procedural-first: store how-to-draw (visual_rpn + transforms); embeddings are secondary/regenerable.
- Hierarchical refs: higher layers reference lower (scene → shapes → strokes → primitives); only include direct primitive refs when not covered by a lower-layer link to reduce graph clutter.
- Dual-client: humans render executed programs; AI consumes procedural programs and embeddings from the same node (extras.k3d).

Suggested ingestion targets:
- Primitive library (already in kernels).
- Stroke/shape/icon libraries (`procedural_drawing_specialist` can emit).
- Scene/layout library (parameterized compositions).
- User-defined gallery (empty by default; writable at runtime for new motifs/scenes).
