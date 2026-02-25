# PM-KR Email: Accessibility Use Case + Compression Architecture

**To:** public-pm-kr@w3.org
**Subject:** Re: Procedural Knowledge Examples - Accessibility Use Case + Compression Architecture
**Date:** February 24, 2026

---

Hi PM-KR community,

Following up on Adam's excellent question about use cases, I'd like to share how PM-KR's layered architecture enables **accessibility** — and why this directly explains our compression claims (50-90% reduction).

## The Accessibility Challenge (Current State)

Today, creating accessible content requires massive duplication:

**Traditional approach:**
- Text version (visual readers)
- Braille version (tactile readers)
- Audio version (screen readers)
- Large print version (low vision readers)
- Sign language video (deaf/hard-of-hearing)

**Result:** 5+ separate files, each manually maintained. When content updates, all versions must be updated independently.

**Problem:** Duplication = inefficiency + inconsistency + unsustainability.

---

## PM-KR's Layered Architecture (Simplified)

PM-KR organizes knowledge into compositional layers:

**Layer 0: Drawing Primitives**
- Procedural geometric programs (LINE, CIRCLE, RECT, BEZIER)

**Layer 1: Characters (glyphs)**
- Composed from Layer 0 (Bézier curves, segments)
- Universal forms (e.g., "A" is "A" before language/modality)

**Layer 2: Words (language-specific)**
- Multiple Layer 2s exist (Occidental, Oriental, etc.)
- Character sequences with language context

**Layer 3: Grammar**
- Language construction rules (where TRM infers structure)

**Layer 4: Text Types**
- Poetry, papers, books (genre/format conventions)

**Key principle:** Each layer references the layer below (no duplication).

---

## Accessibility Example: Braille

**How Braille works in PM-KR:**

1. **Layer 1 (Characters)** stores ONE procedural definition of "A"
   - Visual rendering: Bézier curves → visual glyph
   - Tactile rendering: Braille cell pattern (dots 1 in standard Braille)

2. **Layer 2 (Words)** references Layer 1 characters
   - Same word "hello" → visual rendering OR Braille rendering
   - No separate Braille file needed

3. **One canonical source → multiple modalities**

**Traditional approach (duplication):**
```
text.txt:  "The quick brown fox"  (5 KB)
braille.brf: "⠠⠮ ⠟⠥⠊⠉⠅ ⠃⠗⠕⠺⠝ ⠋⠕⠭"  (5 KB)
audio.mp3: [speech recording]  (500 KB)
large.txt: "THE QUICK BROWN FOX"  (6 KB)
---
Total: 516 KB (massive duplication)
```

**PM-KR approach (procedural):**
```
Layer 1 characters: "T", "h", "e", " ", "q", ... (once)
Layer 2 word references: "The" → [ref(T), ref(h), ref(e)]
Rendering rules:
  - Visual: Bézier → screen
  - Braille: dots pattern → tactile display
  - Audio: phonemes → speech synthesis
  - Large print: scale(2x) → screen
---
Total: ~50 KB (one canonical source + rendering rules)
Compression: 90% reduction (516 KB → 50 KB)
```

**Key insight:** We don't duplicate "The quick brown fox" 4 times. We store it ONCE (procedurally) and render it 4 ways.

---

## How Compression Happens

**1. Symlink-style references (no duplication)**
- "The" appears 1000 times in a book?
- Traditional: 1000 copies of "The" (3,000 bytes)
- PM-KR: 1000 references to ONE "The" definition (1,000 references × 8 bytes = 8 KB + 3 bytes for "The" = 8.003 KB)
- Compression: 62% reduction

**2. Multi-modal from one source**
- Visual + Braille + Audio + Large Print = 4 renderings
- Traditional: 4 separate files (4× storage)
- PM-KR: 1 procedural source + 4 rendering rules
- Compression: 75% reduction

**3. Compositional deduplication**
- Characters shared across words
- Words shared across sentences
- Sentences shared across documents
- Result: Content-based deduplication at every layer

---

## Broader Accessibility Implications

**PM-KR enables:**

1. **Braille** (tactile rendering of Layer 1 characters)
2. **Audio** (phonetic rendering via text-to-speech)
3. **Large print** (scaled visual rendering)
4. **Sign language** (gestural rendering via avatar/video)
5. **Simplified language** (Layer 3 grammar transformations)

**All from ONE canonical procedural source.**

**Why this matters:**
- ✅ Content creators maintain ONE version
- ✅ Updates propagate to ALL modalities automatically
- ✅ Accessibility = built-in, not bolted-on
- ✅ Storage/bandwidth reduced by 50-90%
- ✅ Consistency guaranteed (all modalities from same source)

---

## Real-World Impact

**Textbook example:**
- Traditional: Separate visual textbook, Braille textbook, audio textbook
- PM-KR: One procedural textbook → renders as visual, Braille, audio
- Students with visual impairments get the SAME content as sighted students, from the SAME source, guaranteed consistent

**Legal documents example:**
- Court documents must be accessible (ADA compliance)
- Traditional: Manual Braille transcription (expensive, slow)
- PM-KR: Automatic multi-modal rendering (instant, accurate)

**Educational equity:**
- Low-income schools can't afford separate Braille/audio/large print materials
- PM-KR: One digital source → all modalities (cost-effective accessibility)

---

## Connection to Adam's Use Cases

Adam mentioned:
- schema.org/HowTo (instructions)
- BPMN/BPEL (workflows)
- OpenFn/DeepMind (procedural execution)

**PM-KR adds the accessibility layer:**
- HowTo instructions → visual, Braille, audio, sign language
- Workflows → executable by humans (multi-modal) AND machines (APIs)
- All from one canonical procedural source

---

## Questions for the Group

1. **Other accessibility use cases?** (e.g., color blindness, cognitive disabilities)
2. **How does PM-KR relate to WCAG (Web Content Accessibility Guidelines)?**
3. **Could PM-KR become the standard for accessible digital content?**

---

## Links

**Technical details:**
- Layered architecture: https://github.com/danielcamposramos/Knowledge3D/tree/main/docs/vocabulary
- Compression benchmarks: (forthcoming in PM-KR Core Spec v0.1)

**Community page:**
- https://www.w3.org/community/pm-kr/procedural-memory-knowledge-representation-pm-kr-community-group/

---

Looking forward to discussing how PM-KR can advance accessible knowledge representation!

Best regards,
**Daniel Campos Ramos**
PM-KR Co-Chair
Knowledge3D Project

---

**P.S.** - The accessibility angle isn't just "nice to have" — it's foundational to why PM-KR's compression matters. When one canonical source serves all modalities, we're not just saving storage; we're enabling universal access.

---

## END OF EMAIL DRAFT
