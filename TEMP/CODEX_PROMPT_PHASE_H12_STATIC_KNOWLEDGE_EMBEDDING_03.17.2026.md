# Phase H12: Static Knowledge Embedding — Books Come Alive

**Spec Author:** Claude (Architecture Partner)
**Date:** March 17, 2026
**Depends on:** Phase H10 (behavior activation) COMPLETE, Phase H11 (DOM projection) COMPLETE
**Sovereignty:** Ingestion path (Python export, flexible). I/O path (viewer rendering, flexible).
**Build:** Python: `pytest`. Viewer: `bash /K3D/Knowledge3D.local/envs/viewer-build/build.sh`

---

## Context

The House has 5 books in the Library, each pointing to a `galaxy_ref`:

| Book | galaxy_ref | Content Module | Organizer Stars |
|------|-----------|----------------|----------------|
| Mathematics Primer | `Book/MathematicsPrimer` | `book_content_mathematics.py` | 18 (4 ch, 10 sec, 4 pg) |
| Language Foundations | `Book/LanguageFoundations` | `book_content_language.py` | 18 |
| Physics Handbook | `Book/PhysicsHandbook` | `book_content_physics.py` | 18 |
| Biology Atlas | `Book/BiologyAtlas` | `book_content_biology.py` | 18 |
| Tool Manual | `Book/ToolManual` | `book_content_tools.py` | 18 |

Each book has chapters → sections → pages, with `taxonomy_refs` pointing to real Galaxy entries (`num_0`, `reality_kinematics_position_update_euler`, `grammar_numeric_literal`, etc.) and `grammar_refs` pointing to Grammar Galaxy rules (`sequential_computation`, `rate_application`, etc.).

**Problem:** This content exists in Python but never reaches the viewer. When you click a book, the tablet shows `Load: Book/MathematicsPrimer` — a dead reference. The viewer has no way to resolve `galaxy_ref` to actual content.

**Solution:** Export a companion `house-content.json` alongside `house.glb`. The viewer loads both. When a book is clicked, the ContentApp resolves its `galaxy_ref` and displays the chapter/section/page hierarchy with real references.

---

## Deliverables

### Track A: Content Export Pipeline (Python)
### Track B: Content Loader + Resolver (Viewer)
### Track C: Enhanced ContentApp Book View

---

## Track A: Content Export Pipeline

### A1. Create `knowledge3d/tools/export_house_content.py`

Exports all book content organizer stars + their referenced Galaxy entries into a single JSON file.

```python
"""Export book content and referenced Galaxy entries as companion JSON."""

import json
from pathlib import Path

from knowledge3d.knowledgeverse.book_content_mathematics import MATHEMATICS_PRIMER_ENTRIES
from knowledge3d.knowledgeverse.book_content_language import LANGUAGE_FOUNDATIONS_ENTRIES
from knowledge3d.knowledgeverse.book_content_physics import PHYSICS_HANDBOOK_ENTRIES
from knowledge3d.knowledgeverse.book_content_biology import BIOLOGY_ATLAS_ENTRIES
from knowledge3d.knowledgeverse.book_content_tools import TOOL_MANUAL_ENTRIES
from knowledge3d.knowledgeverse.seed_stars import SEED_STARS
from knowledge3d.knowledgeverse.meaning_star import MeaningCentricStar


def _star_to_content_entry(star: MeaningCentricStar) -> dict:
    """Convert a MeaningCentricStar to a JSON-serializable content entry."""
    entry = {
        "star_id": star.star_id,
        "meaning_class": star.meaning_class,
        "domain": star.domain,
        "meaning_rpn": star.meaning_rpn,
        "behavior_rpn": star.behavior_rpn,
        "surface_forms": {
            lang: {"word_ref": sf.word_ref, "char_refs": sf.char_refs}
            for lang, sf in star.surface_forms.items()
        },
        "taxonomy_refs": list(star.taxonomy_refs),
        "grammar_refs": list(star.grammar_refs),
        "component_refs": list(star.component_refs),
    }
    if star.visual_rpn:
        entry["visual_rpn"] = star.visual_rpn
    return entry


def export_house_content(output_path: Path) -> dict:
    """Export all book content + seed stars to JSON."""
    books = {
        "Book/MathematicsPrimer": MATHEMATICS_PRIMER_ENTRIES,
        "Book/LanguageFoundations": LANGUAGE_FOUNDATIONS_ENTRIES,
        "Book/PhysicsHandbook": PHYSICS_HANDBOOK_ENTRIES,
        "Book/BiologyAtlas": BIOLOGY_ATLAS_ENTRIES,
        "Book/ToolManual": TOOL_MANUAL_ENTRIES,
    }

    content = {
        "version": 1,
        "books": {},
        "concepts": {},
    }

    for galaxy_ref, entries in books.items():
        content["books"][galaxy_ref] = {
            "entries": [_star_to_content_entry(star) for star in entries],
        }

    for star in SEED_STARS:
        content["concepts"][star.star_id] = _star_to_content_entry(star)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(content, indent=2, ensure_ascii=False))

    total_entries = sum(len(b["entries"]) for b in content["books"].values())
    return {
        "books": len(content["books"]),
        "book_entries": total_entries,
        "concepts": len(content["concepts"]),
        "output": str(output_path),
    }


if __name__ == "__main__":
    out = Path("viewer/public/house-content.json")
    result = export_house_content(out)
    print(f"Exported: {result['books']} books, "
          f"{result['book_entries']} entries, "
          f"{result['concepts']} concepts → {result['output']}")
```

### A2. Add to export_house.py

Call `export_house_content()` at the end of the existing `export_house.py` script, so both `house.glb` and `house-content.json` are produced together.

### A3. Update `scripts/export_house.py` wrapper

If there's a wrapper script, have it call both exports. The content JSON should land at `viewer/public/house-content.json` alongside `viewer/public/house.glb`.

---

## Track B: Content Loader + Resolver

### B1. Create `viewer/src/contentLoader.ts`

Loads and indexes the companion content JSON.

```typescript
export interface ContentEntry {
  star_id: string;
  meaning_class: string;
  domain: string;
  meaning_rpn: string;
  behavior_rpn: string;
  surface_forms: Record<string, { word_ref: string; char_refs: string[] }>;
  taxonomy_refs: string[];
  grammar_refs: string[];
  component_refs: string[];
  visual_rpn?: string;
}

export interface BookContent {
  entries: ContentEntry[];
}

export interface HouseContent {
  version: number;
  books: Record<string, BookContent>;
  concepts: Record<string, ContentEntry>;
}

let houseContent: HouseContent | null = null;

export async function loadHouseContent(url: string): Promise<HouseContent> {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`Failed to load ${url}: ${response.status}`);
  houseContent = await response.json();
  return houseContent!;
}

export function getBookContent(galaxyRef: string): BookContent | null {
  return houseContent?.books[galaxyRef] || null;
}

export function getConceptEntry(conceptId: string): ContentEntry | null {
  return houseContent?.concepts[conceptId] || null;
}

export function resolveRef(ref: string): ContentEntry | null {
  // Check concepts first
  const concept = houseContent?.concepts[ref];
  if (concept) return concept;
  // Search across all book entries
  if (!houseContent) return null;
  for (const book of Object.values(houseContent.books)) {
    const entry = book.entries.find(e => e.star_id === ref);
    if (entry) return entry;
  }
  return null;
}

export function getLoadedContent(): HouseContent | null {
  return houseContent;
}
```

### B2. Load content in main.ts

After loading `house.glb`, also load `house-content.json`:

```typescript
import { loadHouseContent } from './contentLoader';

// In loadHouseSceneAsset(), after successful house load:
try {
  await loadHouseContent(url.replace(/\.glb$/, '-content.json').replace(/house\.glb/, 'house-content.json'));
} catch (e) {
  console.warn('House content not available:', e);
}
```

The content load is best-effort — the viewer works without it (falls back to metadata-only display as today).

---

## Track C: Enhanced ContentApp Book View

### C1. Update activator to pass book content

In `activator.ts`, when `handleLoadGalaxy()` fires, resolve the `galaxyRef` to actual content:

```typescript
import { getBookContent } from '../contentLoader';

private handleLoadGalaxy(galaxyRef: string, node: HouseNode): void {
  this.tablet.showFocus();
  this.tablet.dispatch({ type: 'open_app', payload: { id: 'content' } });

  const bookContent = getBookContent(galaxyRef);
  this.tablet.dispatch({
    type: 'showContent',
    payload: bookContent
      ? renderBookContent(galaxyRef, bookContent, node)
      : renderContentPayload({
          node,
          galaxyRef,
          title: node.surfaceForms.en?.word_ref || node.starId,
        }),
  });
}
```

### C2. Create `renderBookContent()` in `contentRenderer.ts`

Renders a book's chapter/section/page hierarchy as a `ContentPage`:

```typescript
import type { BookContent, ContentEntry } from '../contentLoader';

export function renderBookContent(
  galaxyRef: string,
  book: BookContent,
  node: HouseNode,
): ContentPage {
  const title = node.surfaceForms.en?.word_ref || galaxyRef;
  const sections: ContentSection[] = [];

  // Group entries by meaning_class hierarchy
  const chapters = book.entries.filter(e => e.meaning_class === 'chapter');
  const otherEntries = book.entries.filter(e => e.meaning_class !== 'chapter');

  for (const chapter of chapters) {
    const chapterTitle = chapter.surface_forms.en?.word_ref || chapter.star_id;
    const lines: string[] = [];

    // Find sections belonging to this chapter (via component_refs)
    for (const compRef of chapter.component_refs) {
      const child = book.entries.find(e => e.star_id === compRef);
      if (!child) continue;
      const childTitle = child.surface_forms.en?.word_ref || child.star_id;
      const refs = child.taxonomy_refs.slice(0, 4).join(', ');
      lines.push(`${child.meaning_class === 'section' ? '§' : '⁋'} ${childTitle}`);
      if (refs) lines.push(`  refs: ${refs}`);
      if (child.grammar_refs.length) {
        lines.push(`  rules: ${child.grammar_refs.join(', ')}`);
      }
    }

    sections.push({ heading: chapterTitle, lines });
  }

  // Add any non-chapter entries that weren't referenced
  const referencedIds = new Set(chapters.flatMap(c => c.component_refs));
  const unreferenced = otherEntries.filter(e => !referencedIds.has(e.star_id));
  if (unreferenced.length) {
    sections.push({
      heading: 'Additional Entries',
      lines: unreferenced.map(e => {
        const name = e.surface_forms.en?.word_ref || e.star_id;
        return `${e.meaning_class}: ${name}`;
      }),
    });
  }

  // Summary section
  sections.push({
    heading: 'Book Info',
    lines: [
      `Galaxy: ${galaxyRef}`,
      `Entries: ${book.entries.length}`,
      `Chapters: ${chapters.length}`,
      `References: ${new Set(book.entries.flatMap(e => e.taxonomy_refs)).size} unique`,
    ],
  });

  return { title, sections };
}
```

### C3. Reference resolution in ContentApp overlay

When displaying a content page in the overlay, taxonomy_refs that point to known concepts should show resolved names instead of raw IDs:

```typescript
// In ContentApp.openOverlay(), when rendering reference lines:
import { resolveRef } from '../contentLoader';

// For each ref line like "→ concept_mathematics":
const resolved = resolveRef(refId);
const displayName = resolved?.surface_forms.en?.word_ref || refId;
// Show: "→ Mathematics (concept_mathematics)" instead of "→ concept_mathematics"
```

### C4. DOM projection for book content

The existing `buildNodeDomProgram()` from H11 already works with `ContentPage`. When book content is loaded, the DOM projection preview will automatically render the richer content (chapters, sections, references) instead of the minimal metadata view.

---

## Tips for Codex

**Tip 1 — Companion JSON, not bigger GLB.** Don't embed content in the GLB. The GLB is geometry + per-node metadata. The content JSON is knowledge data. Separation of concerns: GLB = spatial structure, JSON = knowledge content.

**Tip 2 — Best-effort content loading.** The viewer must work without `house-content.json` (graceful degradation). The existing metadata-only display (H10) is the fallback. Content loading enhances, never gates.

**Tip 3 — Content entry schema mirrors MeaningStar.** The `ContentEntry` TypeScript interface is a projection of Python `MeaningCentricStar`. Same field names, same structure. The JSON is the bridge.

**Tip 4 — Book content rendering is hierarchical.** Chapters contain component_refs → sections/pages. Render as nested structure: chapter heading → indented sections → refs. Don't flatten.

**Tip 5 — Reference resolution is lookup, not fetch.** `resolveRef()` checks in-memory content. No network calls. The entire content JSON is loaded once at startup.

**Tip 6 — Content file naming.** `house-content.json` alongside `house.glb` in `viewer/public/`. The URL derivation is simple: same directory, different name.

**Tip 7 — Export both files together.** Update `export_house.py` to call `export_house_content()` at the end. One export command produces both files.

---

## Tests

### Python: `tests/test_export_house_content.py`

```python
def test_export_house_content_produces_valid_json():
    from knowledge3d.tools.export_house_content import export_house_content
    import tempfile, json
    from pathlib import Path
    with tempfile.TemporaryDirectory() as tmpdir:
        out = Path(tmpdir) / "content.json"
        result = export_house_content(out)
        assert result["books"] == 5
        assert result["book_entries"] == 90  # 5 books × 18 entries
        assert result["concepts"] == 10
        data = json.loads(out.read_text())
        assert data["version"] == 1
        assert "Book/MathematicsPrimer" in data["books"]

def test_book_entries_have_required_fields():
    from knowledge3d.tools.export_house_content import export_house_content
    import tempfile, json
    from pathlib import Path
    with tempfile.TemporaryDirectory() as tmpdir:
        out = Path(tmpdir) / "content.json"
        export_house_content(out)
        data = json.loads(out.read_text())
        for galaxy_ref, book in data["books"].items():
            for entry in book["entries"]:
                assert "star_id" in entry
                assert "meaning_class" in entry
                assert "surface_forms" in entry
                assert "taxonomy_refs" in entry
```

### Viewer: `viewer/tests/contentLoader.test.ts`

```typescript
import { resolveRef, getBookContent } from '../src/contentLoader';

// Mock fetch for testing
describe('content loader', () => {
  it('resolves known concept ref', () => {
    // After loading mock content...
    // expect(resolveRef('concept_mathematics')?.star_id).toBe('concept_mathematics');
  });

  it('returns null for unknown ref', () => {
    expect(resolveRef('nonexistent_star')).toBeNull();
  });
});
```

### Non-regression

All existing tests must pass. Build via `build.sh`. Export via `export_house.py`.

---

## Success Criteria

1. `python3 knowledge3d/tools/export_house_content.py` produces `viewer/public/house-content.json`
2. Content JSON contains 5 books × 18 entries = 90 book entries + 10 concept stars
3. Viewer loads `house-content.json` alongside `house.glb`
4. Clicking a book shows chapter/section/page hierarchy on tablet (not just "Load: ...")
5. Taxonomy refs show resolved names when concept data is available
6. DOM projection preview renders book content hierarchy
7. Viewer still works if `house-content.json` is missing (graceful degradation)
8. All Python + viewer tests pass, TypeScript clean, build succeeds

---

## Files Changed/Created

| File | Action |
|------|--------|
| `knowledge3d/tools/export_house_content.py` | **NEW** — Content JSON exporter |
| `scripts/export_house.py` | Call content export after GLB export |
| `viewer/public/house-content.json` | **NEW** — Generated content data |
| `viewer/src/contentLoader.ts` | **NEW** — Content JSON loader + resolver |
| `viewer/src/behavior/activator.ts` | Resolve galaxy_ref to book content |
| `viewer/src/behavior/contentRenderer.ts` | Add `renderBookContent()` |
| `viewer/src/apps.ts` | Reference resolution in ContentApp overlay |
| `viewer/src/main.ts` | Load house-content.json at startup |
| `tests/test_export_house_content.py` | **NEW** |
| `viewer/tests/contentLoader.test.ts` | **NEW** |

---

## Architectural Note

This phase makes the books **real**. Every book on the Library shelf is now a consultable reference — click it and see its chapters, sections, and the actual Galaxy entries it references. The Knowledge Tree branches, Gallery displays, Workshop tools — all have taxonomy_refs that resolve to named concepts.

The pattern is: **export once (Python), load once (viewer), resolve instantly (in-memory)**. No backend needed for static content. The companion JSON is the knowledge payload; the GLB is the spatial structure. Together they form a complete, self-contained, browsable knowledge house.

This directly serves the MVP: open browser → navigate House → click book → read real content. No server required. Just files.
