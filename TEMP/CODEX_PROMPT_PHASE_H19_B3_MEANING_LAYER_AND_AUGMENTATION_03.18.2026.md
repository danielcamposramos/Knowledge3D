# Phase H19 + B3 — Multilingual Meaning Layer + Benchmark Knowledge Augmentation

**Two phases, one delivery. H19 is the foundation. B3 builds on it.**

**Creates:**
- `knowledge3d/ingestion/universal_knowledge/multilingual_meanings.py` (H19)
- `knowledge3d/tools/knowledge_proceduralizer.py` (B3)
- `tests/test_multilingual_meanings.py`
- `tests/test_knowledge_proceduralizer.py`

**Modifies:**
- `knowledge3d/ingestion/universal_knowledge/__init__.py`

**Data Sources:**
- OMW: `/K3D/K3D_llama_cpp/datasets/omw-data/omw-data-main/wns/` (31 languages, 117K synsets)
- MMLU: `/K3D/K3D_llama_cpp/datasets/MMLU/data/` (val + auxiliary_train)
- GSM8K: `/K3D/K3D_llama_cpp/datasets/GSM8K/` (train.jsonl)

---

# PART 1: H19 — Multilingual Meaning Layer (NO LLM needed)

## Goal

Build the MEANING LAYER from Open Multilingual Wordnet. One MeaningCentricStar per synset. Surface_forms from all available languages as symlinks. This is pure data transformation — no Ollama required.

## Critical: Semantic Gravity Operates BETWEEN Stars

**Each MeaningCentricStar is already multilingual** — all surface_forms live inside ONE star. "water" (en) and "água" (pt) are the SAME star, not two stars attracting each other.

**Semantic gravity** (`F = T(s₁,s₂) × M(s₁) × M(s₂) / d²`) operates BETWEEN **different** meaning stars based on semantic proximity of their MEANINGS. For example: the star for "water" and the star for "liquid" attract because their meanings are close. Language is irrelevant to the force — only meaning distance matters.

**Design consequence:** Never model surface_forms as separate stars that "orbit" a meaning center. Each star IS the meaning center, carrying all its language representations internally. The gravitational force clusters semantically related CONCEPTS together in Galaxy working memory.

## Critical Language Rule

**English is ALWAYS the primary language (W3C standard).**

- `meaning_rpn`: ALWAYS in English
- Definitions/semantic content: ALWAYS in English
- Other languages appear ONLY as `surface_forms` symlink references (word_ref + char_refs)
- **Single exception:** If a concept has NO English equivalent at all (the synset has zero English lemmas), THEN the meaning text uses the source language. Mark with `LANG_{XX}` prefix in RPN. These are rare culture-specific concepts.

## OMW Data Format

Tab-separated files at `wns/{lang}/wn-data-{lang}.tab`:

```
# Comment line
{synset_id}\tlemma\t{word}              ← word entry (eng file)
{synset_id}\t{lang}:lemma\t{word}       ← word entry (non-eng files)
{synset_id}\t{lang}:def\t{n}\t{text}    ← definition (n = index)
{synset_id}\t{lang}:exe\t{n}\t{text}    ← example sentence
```

Synset IDs are GLOBALLY SHARED: `00001740-a` in `eng/` = same concept in `por/`, `fra/`, `jpn/`, etc.

POS suffix: `n` = noun, `v` = verb, `a` = adjective, `r` = adverb.

Available languages (31 dirs, use the ones that have wn-data-*.tab):
```
als arb bul cwn cow dan ell eng fas fin fra heb hrv isl ita iwn jpn mcr msa nld nor pol por ron slk slv swe tha wikt cldr
```

Language code mapping (OMW dir → ISO 639-1 for surface_forms):
```python
OMW_LANG_MAP = {
    "eng": "en", "por": "pt", "fra": "fr", "jpn": "ja", "arb": "ar",
    "ita": "it", "dan": "da", "ell": "el", "fin": "fi", "heb": "he",
    "hrv": "hr", "isl": "is", "nld": "nl", "nor": "no", "pol": "pl",
    "ron": "ro", "slk": "sk", "slv": "sl", "swe": "sv", "tha": "th",
    "bul": "bg", "fas": "fa", "msa": "ms",
    "als": "sq", "cwn": "zh", "iwn": "id", "mcr": "es", "cow": "zh",
    # Skip: "wikt" and "cldr" are multilingual meta-sources, not single languages
}
```

**Note:** Both `cwn` and `cow` map to `zh` — merge their lemmas under the same `zh` surface_form.

## New File: `knowledge3d/ingestion/universal_knowledge/multilingual_meanings.py`

### Core data structures

```python
@dataclass
class SynsetEntry:
    """One synset with lemmas merged from all languages."""
    synset_id: str                                  # e.g., "00001740-a"
    pos: str = ""                                   # n, v, a, r
    lemmas: dict[str, list[str]] = field(default_factory=dict)    # lang_code → [lemma, ...]
    definitions: dict[str, str] = field(default_factory=dict)     # lang_code → definition text
    examples: dict[str, list[str]] = field(default_factory=dict)  # lang_code → [example, ...]
```

### Functions to implement

**`parse_omw_tab(filepath: Path, lang_code: str) -> dict[str, SynsetEntry]`**
- Parse one `wn-data-{lang}.tab` file
- Handle both `lemma` (eng style) and `{lang}:lemma` (non-eng style) field formats
- Extract lemmas, definitions, examples
- Validate synset_id format: `^\d{8}-[nvar]$`
- Skip comment lines (starting with `#`)

**`load_all_omw(omw_path: Path | None = None) -> dict[str, SynsetEntry]`**
- Iterate all language dirs in `wns/`
- Call `parse_omw_tab()` for each
- Merge: same synset_id accumulates lemmas from all languages into one SynsetEntry
- Skip `wikt` and `cldr` dirs (multilingual meta-sources)
- When two dirs map to same language (cwn + cow → zh), merge their lemmas

**`synset_to_star(entry: SynsetEntry) -> MeaningCentricStar`**
- `star_id`: `"synset_{synset_id}"` with dash → underscore (e.g., `synset_00001740_a`)
- `meaning_class`: POS map: n→"noun", v→"verb", a→"adjective", r→"adverb"
- `meaning_rpn`: **ENGLISH-PRIMARY RULE:**
  - If English definition exists: `"SYNSET {POS} {EN_LEMMA} DEF {en_def[:80]}"`
  - If English lemma but no definition: `"SYNSET {POS} {EN_LEMMA}"`
  - If NO English at all (culture-specific): `"SYNSET {POS} LANG_{XX} {LEMMA} DEF {def[:80]}"` using first available language
- `surface_forms`: One per language. Use FIRST lemma as `word_ref`, `char_refs` via `_house_utils.char_refs(lemma, lang)`
- `taxonomy_refs`: `["concept_language", "wordnet_synset", "concept_{pos}"]`
- `meta_refs`: `["wordnet:{synset_id}", "languages:{count}"]` + up to 20 synonym refs `"synonym:{lang}:{lemma}"` for extra lemmas beyond the first per language
- `house_room`: `"House/Library"`
- `domain`: `"Foundation/Language"`
- `confidence`: 1, `polarity`: 1

**`iter_meaning_stars(omw_path, *, min_languages, pos_filter, limit) -> Iterator[MeaningCentricStar]`**
- Yield stars filtered by min language coverage and POS
- Sorted by synset_id for deterministic output

**`build_meaning_layer_stars(omw_path, *, min_languages=3, limit=None) -> list[MeaningCentricStar]`**
- Convenience wrapper returning a list

**`meaning_layer_stats(stars) -> dict`**
- total_stars, total_surface_forms, avg_languages_per_star, languages_covered, top_languages, pos_distribution

### Update `__init__.py`

Add imports and `__all__` entries for all new public symbols from `multilingual_meanings`.

## Expected output example

Synset `00001740-a` ("able/capable"):

```python
MeaningCentricStar(
    star_id="synset_00001740_a",
    meaning_class="adjective",
    meaning_rpn="SYNSET A ABLE DEF having the necessary means or skill or know-how or authority to do",
    domain="Foundation/Language",
    taxonomy_refs=["concept_language", "wordnet_synset", "concept_adjective"],
    surface_forms={
        "en": SurfaceForm(word_ref="en_able", char_refs=["char_a", "char_b", "char_l", "char_e"]),
        "pt": SurfaceForm(word_ref="pt_capaz", char_refs=["char_c", "char_a", "char_p", "char_a", "char_z"]),
        "ja": SurfaceForm(word_ref="ja_可能", char_refs=["char_ja_u53ef", "char_ja_u80fd"]),
        "ar": SurfaceForm(word_ref="ar_قادر", char_refs=["char_ar_u0642", ...]),
        "fi": SurfaceForm(word_ref="fi_kykenevä", char_refs=[...]),
        "it": SurfaceForm(word_ref="it_abile", char_refs=[...]),
        ...
    },
    meta_refs=[
        "wordnet:00001740-a", "languages:8",
        "synonym:fi:pystyvä", "synonym:fi:taitava", "synonym:it:intelligente",
    ],
    house_room="House/Library",
    confidence=1, polarity=1,
)
```

**Key:** meaning_rpn is ENGLISH. Surface_forms are multilingual symlinks. char_refs trace to Layer 1 glyphs.

## H19 Tests: `tests/test_multilingual_meanings.py`

```python
OMW_PATH = Path("/K3D/K3D_llama_cpp/datasets/omw-data/omw-data-main/wns")
HAS_OMW = OMW_PATH.exists()

@pytest.mark.skipif(not HAS_OMW, reason="OMW data not available")
def test_parse_omw_tab_english():
    """English tab has 200K+ lemmas, includes synset 00001740-a with 'able'."""
    synsets = parse_omw_tab(OMW_PATH / "eng" / "wn-data-eng.tab", "en")
    assert len(synsets) > 1000
    assert "able" in synsets["00001740-a"].lemmas["en"]

@pytest.mark.skipif(not HAS_OMW, reason="OMW data not available")
def test_parse_omw_tab_portuguese():
    synsets = parse_omw_tab(OMW_PATH / "por" / "wn-data-por.tab", "pt")
    assert "capaz" in synsets["00001740-a"].lemmas["pt"]

@pytest.mark.skipif(not HAS_OMW, reason="OMW data not available")
def test_load_all_omw_merges():
    synsets = load_all_omw(OMW_PATH)
    assert len(synsets) > 10000
    entry = synsets["00001740-a"]
    assert len(entry.lemmas) >= 3  # en + pt + at least one more

@pytest.mark.skipif(not HAS_OMW, reason="OMW data not available")
def test_synset_to_star_english_primary():
    """meaning_rpn is ALWAYS English."""
    synsets = load_all_omw(OMW_PATH)
    star = synset_to_star(synsets["00001740-a"])
    assert star.star_id == "synset_00001740_a"
    assert "ABLE" in star.meaning_rpn  # English, not Portuguese/Japanese
    assert "en" in star.surface_forms
    assert star.surface_forms["en"].word_ref == "en_able"

@pytest.mark.skipif(not HAS_OMW, reason="OMW data not available")
def test_surface_forms_have_char_refs():
    """Every surface_form must have char_refs (Layer 1 symlinks)."""
    synsets = load_all_omw(OMW_PATH)
    star = synset_to_star(synsets["00001740-a"])
    for lang, sf in star.surface_forms.items():
        assert len(sf.char_refs) > 0, f"Missing char_refs for {lang}"

@pytest.mark.skipif(not HAS_OMW, reason="OMW data not available")
def test_min_languages_filter():
    stars = list(iter_meaning_stars(OMW_PATH, min_languages=5, limit=20))
    for star in stars:
        assert len(star.surface_forms) >= 5

@pytest.mark.skipif(not HAS_OMW, reason="OMW data not available")
def test_pos_filter():
    nouns = list(iter_meaning_stars(OMW_PATH, min_languages=2, pos_filter={"n"}, limit=20))
    assert all(s.meaning_class == "noun" for s in nouns)

@pytest.mark.skipif(not HAS_OMW, reason="OMW data not available")
def test_build_meaning_layer_stats():
    stars = build_meaning_layer_stars(OMW_PATH, min_languages=3, limit=100)
    stats = meaning_layer_stats(stars)
    assert stats["total_stars"] == 100
    assert stats["avg_languages_per_star"] >= 3.0

def test_pos_map_values():
    assert POS_MAP["n"] == "noun"
    assert POS_MAP["v"] == "verb"
    assert POS_MAP["a"] == "adjective"
    assert POS_MAP["r"] == "adverb"

def test_lang_map_key_languages():
    assert OMW_LANG_MAP["eng"] == "en"
    assert OMW_LANG_MAP["por"] == "pt"
    assert OMW_LANG_MAP["jpn"] == "ja"
```

---

# PART 2: B3 — Benchmark Knowledge Augmentation via Ollama

**Depends on:** H19 (meaning layer must exist), H17 (foundation stars), B2c (Ollama chat() API)

## Goal

Use Ollama to proceduralize benchmark TRAINING data into Galaxy stars that SYMLINK to existing H19 meaning stars and H17 foundation stars. The LLM's job is to decompose knowledge into references, not to store knowledge itself.

## Architecture

```
Benchmark Training Data (MMLU CSV, GSM8K JSONL)
    ↓
knowledge_proceduralizer.py
    ↓
For each entry:
    ├─ Build RAG context: existing star_ids from H17 + H19
    │   (elements, constants, units, AND meaning synset_ids)
    ├─ Build user prompt: the training entry content
    ├─ System prompt: explains K3D symlink model, asks for structured JSON
    ├─ Send to Ollama chat() API (qwen3:8b for MMLU, qwen2.5:32b for GSM8K)
    ├─ Parse JSON response → AugmentationResult
    ├─ result_to_star() → MeaningCentricStar
    │   (star.meta_refs contain symlinks to H17/H19 star_ids)
    └─ Accumulate
    ↓
write_stars_jsonl() → /K3D/Knowledge3D.local/galaxies/proceduralized_*.jsonl
```

## System Prompt

```python
PROCEDURALIZATION_SYSTEM_PROMPT = """You are a knowledge proceduralizer for K3D, a spatial knowledge system.

Your job: extract MEANING from educational content and convert it to structured procedural entries that REFERENCE existing knowledge — never duplicate it.

## K3D Knowledge Layers
1. Form (Layer 1): How something looks — glyphs, shapes, visual primitives
2. Meaning (Layer 2): What something IS — language-agnostic concept. One star per meaning.
3. Rules (Layer 3): How things relate — formulas, grammar rules, transformations
4. Meta-Rules (Layer 4): Rules about rules — domain constraints, when to apply which rule

## Symlink Principle (CRITICAL)
NEVER restate a fact that already has a star_id. REFERENCE it instead.
- Chemistry fact about carbon? → Reference "element_c" (NOT "Carbon has atomic number 6")
- Physics uses speed of light? → Reference "constant_speed_of_light" (NOT "c = 299792458 m/s")
- Word "water" in the content? → Reference "synset_14845743_n" (NOT inline the definition)

## Your Output Format
Return strict JSON:
{
  "meaning_class": "fact|rule|formula|definition|pattern",
  "meaning_rpn": "compact English RPN (e.g., ELEMENT_C ATOMIC_NUMBER 6 HAS_PROPERTY)",
  "domain": "Mathematics|Physics|Biology|Language|Tools|General",
  "summary": "one-line English factual summary",
  "star_refs": ["element_c", "synset_14845743_n", "constant_speed_of_light"],
  "entities": [{"name": "carbon", "star_ref": "element_c"}, {"name": "water", "star_ref": "synset_14845743_n"}],
  "relationships": [{"from": "entity", "relation": "is_a|has_property|causes|requires|composed_of", "to": "entity"}],
  "taxonomy_refs": ["concept_chemistry", "periodic_table"],
  "surface_forms": {"en": "english label"},
  "grammar_rules": [{"pattern": "IF condition THEN result", "strength": 1}],
  "layer": 2,
  "confidence": 0.0-1.0
}

Key rules:
- meaning_rpn MUST be in English, using RPN notation
- star_refs: list ALL existing star_ids this entry connects to (this is the symlink graph)
- entities: tag each entity with its star_ref if one exists in the reference list
- grammar_rules: only for Layer 3+ entries (rules, formulas, patterns)
- Be PRECISE. No narrative. Every field matters."""
```

## RAG Context Builder

The RAG gives Ollama a menu of EXISTING star_ids it can symlink to. Two sources:

1. **H17 foundation stars** — elements, constants, units, materials, formats
2. **H19 meaning stars** — synset_ids for common words

```python
def build_rag_context(domain: str, subject: str, question_text: str) -> str:
    """Build compact reference of existing star_ids for Ollama to symlink to."""

    refs: list[str] = []
    refs.append("## Existing star_ids (REFERENCE these, do not restate their content):")
    refs.append("")

    combined = f"{domain} {subject} {question_text}".lower()

    # --- H17 Foundation Stars ---

    # Always show taxonomy concepts
    refs.append("### Taxonomy")
    refs.append("concept_mathematics, concept_physics, concept_chemistry, concept_biology, concept_language, concept_tool")

    # Chemistry: show relevant elements
    if _hits(combined, {"chem", "element", "atom", "molecule", "compound", "reaction",
                         "oxide", "acid", "metal", "halogen", "periodic", "bond",
                         "bio", "anatomy", "medicine", "organic", "cell"}):
        from knowledge3d.ingestion.universal_knowledge import iter_elements
        elements = list(iter_elements())[:36]
        refs.append("")
        refs.append("### Chemical elements (star_id = element_{symbol})")
        for el in elements:
            refs.append(f"  element_{el.symbol.lower()} = {el.name_en}, Z={el.atomic_number}, mass={el.atomic_mass}")

    # Physics: show constants
    if _hits(combined, {"phys", "force", "energy", "velocity", "gravity", "light",
                         "planck", "boltzmann", "electric", "magnetic", "thermo",
                         "momentum", "wave", "frequency", "astro", "optic"}):
        from knowledge3d.ingestion.universal_knowledge import iter_physical_constants
        refs.append("")
        refs.append("### Physical constants (star_id = constant_{key})")
        for c in iter_physical_constants():
            refs.append(f"  constant_{c.key} = {c.name} = {c.value} {c.unit}")

    # Math/measurements: show unit domains
    if _hits(combined, {"math", "algebra", "calculus", "geometry", "unit", "convert",
                         "distance", "speed", "mass", "temperature", "pressure",
                         "econ", "statistic", "probability"}):
        from knowledge3d.ingestion.universal_knowledge import iter_domains
        refs.append("")
        refs.append("### Measurement units (star_id = unit_{domain}_{unit})")
        for d in list(iter_domains())[:8]:
            units = list(d.units.keys())[:4]
            refs.append(f"  {d.key}: " + ", ".join(f"unit_{d.key}_{u}" for u in units))

    # Materials
    if _hits(combined, {"material", "steel", "glass", "water", "wood", "concrete", "alloy"}):
        refs.append("")
        refs.append("### Materials (star_id = material_{name})")
        refs.append("  material_water, material_steel, material_glass, material_wood, material_concrete")

    # --- H19 Meaning Stars ---
    # For any domain, show a sample of relevant synset_ids so the model knows they exist
    refs.append("")
    refs.append("### Word meanings (star_id = synset_{id}, one star per meaning, multilingual)")
    refs.append("  Thousands available. Reference format: synset_XXXXXXXX_X")
    refs.append("  If you recognize a concept that would have a WordNet synset, reference it as synset_{probable_id}")
    refs.append("  The system will resolve or create as needed.")

    # H17 prefixes summary
    refs.append("")
    refs.append("### All available star_id prefixes:")
    refs.append("  element_*, constant_*, unit_*, material_*, script_*, numeral_system_*,")
    refs.append("  format_*, standard_size_*, synset_*")

    return "\n".join(refs)


def _hits(text: str, keywords: set[str]) -> bool:
    return any(kw in text for kw in keywords)
```

## Data Loaders

### MMLU loader

```python
def load_mmlu_entries(
    data_dir: Path,
    split: str = "val",
    *,
    subjects: list[str] | None = None,
    limit_per_subject: int | None = None,
) -> Iterator[dict[str, Any]]:
    """Load MMLU Q&A pairs as knowledge entries to proceduralize."""
```

- Read CSV from `data_dir / split / *_{split}.csv`
  - val split: files named `{subject}_val.csv`
  - auxiliary_train split: files named `{subject}.csv` (no suffix)
  - test split: files named `{subject}_test.csv`
- For each row: `[question, optA, optB, optC, optD, correct_letter]`
- Format content as:
  ```
  Subject: {subject title-cased}
  Question: {question}
  Correct Answer: {correct_letter}. {correct_answer_text}
  Key Fact: The answer to "{question}" is "{correct_answer_text}".
  ```
- Yield dict with: `entry_id`, `content`, `subject`, `domain_hint`, `source`, `correct_answer`, `question`

### GSM8K loader

```python
def load_gsm8k_entries(data_dir: Path, *, limit: int | None = None) -> Iterator[dict[str, Any]]:
    """Load GSM8K train problems as arithmetic pattern knowledge."""
```

- Read JSONL from `data_dir / grade_school_math / data / train.jsonl`
- Each line: `{"question": "...", "answer": "step-by-step\n#### final_number"}`
- Format content as:
  ```
  Subject: Grade School Mathematics
  Problem: {question}
  Step-by-step Solution: {answer_text}
  Final Answer: {final_number}
  Extract the arithmetic PATTERN and RULES used in this solution.
  ```
- Yield dict with: `entry_id`, `content`, `subject="arithmetic"`, `domain_hint="Mathematics"`, `source="gsm8k_train"`

### Subject → domain mapping

```python
def _subject_to_domain(subject: str) -> str:
    s = subject.lower()
    if any(kw in s for kw in ["math", "algebra", "calculus", "geometry", "statistics"]):
        return "Mathematics"
    if any(kw in s for kw in ["physics", "astronomy", "electrical"]):
        return "Physics"
    if any(kw in s for kw in ["chemistry"]):
        return "Physics"  # Chemistry lives in Reality Galaxy alongside Physics
    if any(kw in s for kw in ["biology", "anatomy", "medicine", "nutrition", "clinical"]):
        return "Biology"
    if any(kw in s for kw in ["computer", "machine_learning", "security"]):
        return "Tools"
    return "General"
```

## Model Routing

```python
SOURCE_MODEL_MAP = {
    "mmlu_train": "qwen3:8b",     # Factual Q&A → fast extraction
    "mmlu_val": "qwen3:8b",
    "gsm8k_train": "qwen2.5:32b", # Word problems → needs reasoning to extract patterns
}

MODEL_OPTIONS = {
    "qwen3:8b": {"temperature": 0.1, "num_predict": 4096},
    "qwen2.5:32b": {"temperature": 0.2, "num_predict": 2048},
}
```

## Core Proceduralization

```python
def proceduralize_entry(entry, ollama, model, options, timeout) -> AugmentationResult | None:
    """Send one entry through Ollama and parse structured JSON response."""
    rag = build_rag_context(entry["domain_hint"], entry["subject"], entry.get("question", ""))
    user_message = f"{rag}\n\n---\n\nProceduralize this knowledge entry:\n\n{entry['content']}"
    result = ollama.chat(model, [
        {"role": "system", "content": PROCEDURALIZATION_SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ], timeout=timeout, temperature=..., options=...)
    # Parse JSON from response (handle <think> blocks, markdown fences, embedded JSON)
    return _parse_response(result.output, entry)
```

**JSON extraction must handle:**
1. Clean JSON: `{"meaning_class": "fact", ...}`
2. Markdown fenced: `` ```json\n{...}\n``` ``
3. Embedded in text: `Here is the result: {...}`
4. qwen3 thinking blocks: `<think>...</think>` before the JSON

**Response parsing → AugmentationResult:**
- Map `star_refs` from response into `meta_refs` on the AugmentationResult
- Map `meaning_rpn` → `meaning_rpn_hint`
- Map `grammar_rules` → store as JSON string in a meta_ref
- Fallback: if JSON parse fails, create minimal AugmentationResult with low confidence

## Batch Runner

```python
def proceduralize_dataset(
    entries: Iterator[dict],
    *, model, timeout, output_path: Path, batch_size=50,
) -> dict:
    """Run batch proceduralization. Print progress. Write stars JSONL."""
```

- For each entry: call `proceduralize_entry()` → `result_to_star()` (from content_to_stars.py)
- Add `meta_refs`: `[f"source:{entry['source']}", f"subject:{entry['subject']}"]`
- Plus all `star_refs` from the Ollama response as `meta_refs` (the symlink graph)
- Write to JSONL via `write_stars_jsonl()`
- Return summary report

## CLI

```python
def main(argv=None):
    """CLI entry point.

    Usage:
        python -m knowledge3d.tools.knowledge_proceduralizer \
            --source mmlu_val --count 20 --subjects astronomy,college_physics
        python -m knowledge3d.tools.knowledge_proceduralizer \
            --source gsm8k_train --count 50 --timeout 300
        python -m knowledge3d.tools.knowledge_proceduralizer \
            --source mmlu_train --count 100 --limit-per-subject 5
    """
```

Arguments:
- `--source`: `mmlu_val`, `mmlu_train`, `gsm8k_train`
- `--count`: max total entries (default 50)
- `--subjects`: comma-separated MMLU subjects (default: all)
- `--limit-per-subject`: max per MMLU subject
- `--model`: override Ollama model
- `--timeout`: per-query timeout (default 120, use 300 for qwen2.5:32b)
- `--output`: JSONL output path (default `/K3D/Knowledge3D.local/galaxies/proceduralized_stars.jsonl`)
- `--mmlu-data`: MMLU data dir (default `/K3D/K3D_llama_cpp/datasets/MMLU/data`)
- `--gsm8k-data`: GSM8K data dir (default `/K3D/K3D_llama_cpp/datasets/GSM8K`)

## B3 Tests: `tests/test_knowledge_proceduralizer.py`

```python
MMLU_PATH = Path("/K3D/K3D_llama_cpp/datasets/MMLU/data")
GSM8K_PATH = Path("/K3D/K3D_llama_cpp/datasets/GSM8K")
HAS_MMLU = (MMLU_PATH / "val").exists()
HAS_GSM8K = (GSM8K_PATH / "grade_school_math" / "data" / "train.jsonl").exists()

@pytest.mark.skipif(not HAS_MMLU, reason="MMLU data not available")
def test_load_mmlu_val_entries():
    entries = list(load_mmlu_entries(MMLU_PATH, "val", subjects=["astronomy"], limit_per_subject=2))
    assert len(entries) >= 1
    assert "Correct Answer:" in entries[0]["content"]
    assert entries[0]["subject"] == "astronomy"

@pytest.mark.skipif(not HAS_GSM8K, reason="GSM8K data not available")
def test_load_gsm8k_entries():
    entries = list(load_gsm8k_entries(GSM8K_PATH, limit=2))
    assert len(entries) >= 1
    assert "Step-by-step" in entries[0]["content"]
    assert entries[0]["domain_hint"] == "Mathematics"

def test_rag_context_chemistry():
    ctx = build_rag_context("Physics", "college_chemistry", "What is the atomic number of carbon?")
    assert "element_" in ctx

def test_rag_context_physics():
    ctx = build_rag_context("Physics", "astronomy", "What is the speed of light?")
    assert "constant_" in ctx

def test_rag_context_always_has_taxonomy():
    ctx = build_rag_context("General", "philosophy", "What is virtue?")
    assert "concept_" in ctx
    assert "synset_" in ctx

def test_rag_context_has_h19_reference():
    ctx = build_rag_context("General", "any", "any question")
    assert "synset_" in ctx  # H19 meaning stars always referenced

def test_extract_json_clean():
    result = _extract_json('{"meaning_class": "fact", "domain": "Physics"}')
    assert result["meaning_class"] == "fact"

def test_extract_json_fenced():
    result = _extract_json('Result:\n```json\n{"meaning_class": "rule"}\n```')
    assert result["meaning_class"] == "rule"

def test_extract_json_with_thinking():
    raw = '<think>analysis...</think>\n{"meaning_class": "fact", "summary": "test"}'
    result = _parse_response(raw, {"domain_hint": "General", "subject": "test"})
    assert result.summary == "test"

def test_subject_to_domain():
    assert _subject_to_domain("college_physics") == "Physics"
    assert _subject_to_domain("abstract_algebra") == "Mathematics"
    assert _subject_to_domain("college_biology") == "Biology"
    assert _subject_to_domain("world_religions") == "General"

def test_system_prompt_has_symlink_principle():
    assert "symlink" in PROCEDURALIZATION_SYSTEM_PROMPT.lower() or "REFERENCE" in PROCEDURALIZATION_SYSTEM_PROMPT
    assert "star_refs" in PROCEDURALIZATION_SYSTEM_PROMPT
    assert "English" in PROCEDURALIZATION_SYSTEM_PROMPT
```

---

# File Changes Summary

| File | Action | Phase |
|------|--------|-------|
| `knowledge3d/ingestion/universal_knowledge/multilingual_meanings.py` | **NEW** | H19 |
| `knowledge3d/ingestion/universal_knowledge/__init__.py` | **MODIFY** | H19 |
| `knowledge3d/tools/knowledge_proceduralizer.py` | **NEW** | B3 |
| `tests/test_multilingual_meanings.py` | **NEW** | H19 |
| `tests/test_knowledge_proceduralizer.py` | **NEW** | B3 |

---

# Execution Order

1. **H19 first** — build `multilingual_meanings.py`, run its tests, verify OMW parsing works
2. **B3 second** — build `knowledge_proceduralizer.py`, it imports from H19 for RAG context
3. **Do NOT run live Ollama** — just build + test (mock Ollama in tests). Live runs are a separate step.

---

# Success Criteria

**H19:**
1. Parses English OMW tab (200K+ lemmas) correctly
2. Merges 31 language sources into unified synset dict
3. Stars have English-primary meaning_rpn
4. Surface_forms have char_refs (Layer 1 symlinks)
5. `build_meaning_layer_stars(min_languages=3, limit=100)` returns 100 stars with 3+ languages
6. 10 tests pass

**B3:**
1. MMLU loader reads val + auxiliary_train CSVs correctly
2. GSM8K loader reads train.jsonl with step-by-step solutions
3. RAG context includes H17 + H19 star_ids
4. System prompt explains symlink model and requests star_refs
5. JSON extraction handles thinking blocks + fences + embedded
6. 11 tests pass

**Combined:** All existing tests non-regression.

---

# Codex: Start Here

**This is your implementation directive. Execute in order:**

## Step 1: H19 — Multilingual Meaning Layer

1. **Create** `knowledge3d/ingestion/universal_knowledge/multilingual_meanings.py`
   - Implement all functions from PART 1 above: `parse_omw_tab`, `load_all_omw`, `synset_to_star`, `iter_meaning_stars`, `build_meaning_layer_stars`, `meaning_layer_stats`
   - The `SynsetEntry` dataclass, `OMW_LANG_MAP`, and `POS_MAP` constants
   - Remember: each star is ONE concept with ALL languages inside (surface_forms dict). Stars are not per-language.
   - English-primary: `meaning_rpn` is ALWAYS English. Other languages are surface_form symlinks only.

2. **Update** `knowledge3d/ingestion/universal_knowledge/__init__.py`
   - Add imports: `from .multilingual_meanings import ...`
   - Add to `__all__`

3. **Create** `tests/test_multilingual_meanings.py`
   - All 10 tests from PART 1 test section
   - Tests that hit OMW data use `@pytest.mark.skipif(not HAS_OMW, ...)`
   - Pure logic tests (POS_MAP, LANG_MAP) run without data

4. **Run tests** — all 10 must pass. Fix until green.

5. **Quick validation** — run `build_meaning_layer_stars(min_languages=3, limit=10)` and print one star to verify:
   - `meaning_rpn` is in English
   - `surface_forms` has 3+ languages
   - Each `surface_form` has `char_refs` pointing to Layer 1

## Step 2: B3 — Benchmark Knowledge Proceduralizer

**Only start after H19 tests pass.**

1. **Create** `knowledge3d/tools/knowledge_proceduralizer.py`
   - All functions from PART 2: data loaders, RAG builder, system prompt, JSON extraction, batch runner, CLI
   - Import from H19 for meaning star references in RAG context
   - Import `OllamaManager` from `knowledge3d.ingestion.ollama_manager` for chat API
   - Import `result_to_star`, `write_stars_jsonl` from `knowledge3d.tools.content_to_stars`

2. **Create** `tests/test_knowledge_proceduralizer.py`
   - All 11 tests from PART 2 test section
   - Tests that hit datasets use `@pytest.mark.skipif`
   - Mock Ollama in unit tests — do NOT make live Ollama calls in tests
   - JSON extraction tests are pure logic (no mocking needed)

3. **Run tests** — all 11 must pass. Fix until green.

4. **Run full test suite** — ensure ALL existing tests still pass (non-regression).

## Step 3: Verify & Report

After both phases pass:

```bash
# Run all tests
python -m pytest tests/test_multilingual_meanings.py tests/test_knowledge_proceduralizer.py -v

# Quick H19 validation
python -c "
from knowledge3d.ingestion.universal_knowledge.multilingual_meanings import build_meaning_layer_stars, meaning_layer_stats
stars = build_meaning_layer_stars(min_languages=3, limit=10)
print(meaning_layer_stats(stars))
print(stars[0])
"
```

Report: files created, tests passing, H19 validation output, any issues.

**Do NOT run live Ollama proceduralization.** That's a separate step after review.
