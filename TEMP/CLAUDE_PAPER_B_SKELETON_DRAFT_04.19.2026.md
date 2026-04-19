# Paper B — Semantic Gravity — Skeleton Draft

**Date**: 2026-04-19
**Authors**: Daniel Campos Ramos (first — idea and formula), Christoph Dorn (second — term coiner), additional PM-KR co-authors TBD
**Target venue**: companion preprint to Paper A (arXiv cs.AI; venue choice TBD)
**Status**: Skeleton — section-by-section targets, formula anchors, example slots.
**Provenance authority**: [`feedback_semantic_gravity_provenance_corrected.md`](../../home/daniel/.claude/projects/-K3D-GitHub-Knowledge3D/memory/feedback_semantic_gravity_provenance_corrected.md). This supersedes `project_semantic_gravity_coinage.md`.
**Related**: [`CLAUDE_PAPER_SERIES_AND_ATTRIBUTIONS_04.18.2026.md`](CLAUDE_PAPER_SERIES_AND_ATTRIBUTIONS_04.18.2026.md), ATTRIBUTIONS.md §4.4.1.

---

## Working Title

**Semantic Gravity: A Ternary Force Law for Meaning-Centric Memory Navigation**

Alternates:
- *Meaning-Mass and the Ternary Gravitational Field of the Galaxy Universe*
- *From Gärdenfors' Conceptual Spaces to a Computable Force Law for Meaning*

---

## Authorship Note (front matter, pre-abstract)

**Provenance of this work (verbatim, to appear as a footnote on the title page):**

> The formula `F = T(s₁,s₂) · M(s₁) · M(s₂) / d²` and the underlying theoretical idea — that meaning-mass and meaning-distance can drive a physical-style force over language-agnostic concept stars — were originated by Daniel Campos Ramos as part of the Knowledge3D architecture. The English-language phrase *"semantic gravity cohered by meaning"* was coined by Christoph Dorn in March 2026 during PM-KR discussion. Authorship order reflects this split: Daniel Campos Ramos as originator of idea and formula, Christoph Dorn as term coiner and co-author.

See ATTRIBUTIONS.md §4.4.1 "Semantic Gravity — Split Provenance".

---

## Abstract (≤ 175 words)

**Target 4-sentence arc:**

1. **Problem.** Memory-retrieval systems treat proximity as a scalar — cosine similarity, dot-product, Euclidean distance — none of which capture that *which* concept is pulling *which* depends on a ternary relation (supporting, neutral, opposing) between them.
2. **Proposal.** We introduce a *semantic gravity* law `F = T(s₁,s₂) · M(s₁) · M(s₂) / d²` in which concept stars have *meaning-mass* `M(·)`, pairwise *meaning-distance* `d(·,·)`, and a *ternary relational operator* `T(·,·) ∈ {-1, 0, +1}`. The law operates over language-agnostic stars, not over surface forms.
3. **Evidence.** We demonstrate the law driving navigation in the K3D Galaxy Universe — a VRAM substrate where ~7M-parameter TRM agents traverse concept space at frame cadence — and show that the ternary operator is load-bearing: cosine-similarity alternatives fail on adversarial pairs where two concepts are close *in surface* but opposed *in meaning*.
4. **Claim.** Semantic gravity is a computable, falsifiable, substrate-level mechanism bridging Gärdenfors-style conceptual spaces and GPU-native cognition.

Word budget: ~170 words.

---

## §1 Introduction (~0.75 page)

### §1.1 Hook (1 paragraph)

*Cosine similarity is a weak instrument. It treats "attract" and "repel" as near-neighbours because they share a physics context, even though the ternary relation between them is `-1`, not `+1`. A force law that ignores sign is not a force law.*

### §1.2 Motivation (1 paragraph)

In a memory-palace cognitive substrate like K3D's Galaxy Universe, the AI agent is constantly choosing *where to look next*. A scalar similarity metric over-binds antonyms and concept-negations; a learned attention head obscures the choice. A physical-style force law over meaning-centric stars is both explicit and ternary.

### §1.3 Contributions (bulleted)

> **B1** — The semantic-gravity force law `F = T(s₁,s₂) · M(s₁) · M(s₂) / d²`, with explicit definitions of `T`, `M`, and `d` over language-agnostic concept stars.
>
> **B2** — A demonstration that the ternary operator `T ∈ {-1, 0, +1}` is load-bearing by adversarial examples where `cos` and `dot-product` fail.
>
> **B3** — A GPU-native implementation over the Galaxy Universe VRAM substrate: the law is evaluated per-pair in the nine-chain swarm at every reasoning tick.
>
> **B4** — A philosophical placement between Gärdenfors' conceptual spaces (theoretical) and K3D's runtime (operational).

### §1.4 Companion paper positioning

Paper A introduces the K3D substrate and its three contributions; Paper B develops the *specific* mechanism that drives Galaxy navigation inside C2 (TRM-as-Avatar). Readers arriving from Paper A can treat §2-§3 as an expansion of Paper A §2.2's "semantic-gravity field over meaning-centric stars" phrase.

---

## §2 Background (~0.75 page)

### §2.1 Conceptual spaces (Gärdenfors 2000)

Gärdenfors' *Conceptual Spaces* proposes meaning as geometry — concepts occupy regions in a quality-dimensional space, and similarity is geometric. **What it lacks.** No explicit force; no sign; no ternary relation; no computable dynamics at runtime.

### §2.2 Cosine similarity and its failure modes

Standard embedding-similarity metrics (cos, dot, Euclidean). Each is scalar-unsigned over the surface form (the embedding). **What they lack.** Cannot encode opposition; cannot encode language-agnostic meaning identity (two languages, same concept, different embeddings).

### §2.3 Physical metaphors in knowledge representation

Spreading activation (Collins & Loftus 1975); force-directed graph layout (Fruchterman-Reingold 1991); latent-space gradient flows. **What they lack.** Metaphors for layout, not substrate-level laws for cognition. No ternary operator.

### §2.4 Ternary and balanced-ternary computation

Knuth TAOCP Vol. 2 §4.1 (balanced-ternary as optimal radix); more recent BitNet-b1.58 weight encoding (Ma et al. 2024). **Why relevant.** These establish that ternary states {-1, 0, +1} are a first-class computational primitive, not just a representational curiosity. Semantic gravity uses the same trit alphabet for `T`.

### §2.5 The gap this paper fills

A computable, signed, language-agnostic, runtime-evaluable force law over concept stars. K3D provides both the formula and a substrate that evaluates it every tick.

---

## §3 The Semantic-Gravity Law (~1.25 pages — the technical core)

### §3.1 Formal statement

For any two stars `s₁, s₂` in the Galaxy Universe:

> **F(s₁, s₂) = T(s₁, s₂) · M(s₁) · M(s₂) / d(s₁, s₂)²**

where:
- **`T(s₁, s₂) ∈ {-1, 0, +1}`** is the *ternary relational operator*: `+1` when `s₂` supports `s₁` in context, `0` when neutral, `-1` when opposing.
- **`M(sᵢ) ≥ 0`** is the *meaning-mass* of star `sᵢ`: a scalar accumulated from multilingual surface forms, usage frequency, cross-galaxy reference count, and RPN-program cardinality. Full definition in §3.3.
- **`d(s₁, s₂) > 0`** is *meaning-distance*: a non-surface metric defined over the canonical meaning-centric axis, not over embeddings of surface forms. Full definition in §3.4.

### §3.2 The ternary operator `T` — core of the novelty

**Why ternary, not scalar.** A scalar similarity answers "how close"; a ternary operator answers "in which direction the force pulls." Two concepts can be semantically close *and* oppositional (attract/repel, good/evil, yes/no). A scalar collapses this; the ternary preserves it.

**How `T` is computed.** `T` is stored per-pair in the Galaxy Universe (not computed from embeddings at query time). Values come from:
- Explicit editorial assertions (Reality Enabler / librarian curation).
- RPN-program composition (if `s₁`'s program calls `s₂`'s program as a subroutine, default `T = +1`).
- Defeasible-logic rule strengths from companion Paper F (SPINdle-derived).

**Worked example.** `T(gravity, momentum) = +1` (both physics, supportive composition). `T(attract, repel) = -1` (oppositional pair). `T(library, database) = 0` (neutral — related but no force in either direction).

### §3.3 Meaning-mass `M`

**Definition.** `M(s) = α · L(s) + β · F(s) + γ · R(s) + δ · P(s)` where:
- `L(s)` = count of languages the star has surface forms in (Matryoshka-embedded per `feedback_use_ollama_specialists.md`'s embedder pipeline);
- `F(s)` = normalised usage frequency across the Knowledgeverse audit log;
- `R(s)` = cross-galaxy reference count (symlink in-degree);
- `P(s)` = RPN-program cardinality (programs that reference this star as a primitive);
- `α, β, γ, δ` are non-negative weights; default values in §3.6 (empirical).

**Why these four.** Each captures one dimension of "how much does this concept *matter* in the Knowledgeverse": breadth of linguistic coverage (L), frequency of use (F), compositional centrality (R), procedural utility (P).

### §3.4 Meaning-distance `d`

**Definition.** `d(s₁, s₂)` is measured in the *meaning-centric* axis of the Galaxy, not in any language's surface embedding space. Specifically: `d(s₁, s₂)² = Σᵢ wᵢ · (qᵢ(s₁) - qᵢ(s₂))²` over Gärdenfors-style quality dimensions `qᵢ` with weights `wᵢ`.

**Why not cosine.** Cosine conflates multilingual surface variants of the same star. K3D's meaning-distance is computed between *canonical stars*, each of which already unifies all language-specific surface forms via bidirectional symlinks (`feedback_bidirectional_symlinks_norm.md`).

### §3.5 Properties of the law

1. **Language-agnostic** — `F` depends on stars, not on words.
2. **Signed** — `T ∈ {-1, 0, +1}` yields attraction, neutrality, or repulsion.
3. **Inverse-square** — `d²` denominator gives natural locality; distant stars exert negligible force.
4. **Computable at tick cadence** — every hop in the Galaxy evaluates `F` per candidate pair.
5. **Falsifiable** — the §4 adversarial test set gives concrete failure modes for scalar-similarity alternatives.

### §3.6 Default parameter choices

Empirical defaults for α, β, γ, δ (to be populated from experiments); the paper should list the actual numbers used in §5 runs.

---

## §4 Adversarial Demonstration — Why Ternary Matters (~0.75 page)

### §4.1 Adversarial pair set

Five hand-crafted star pairs where cosine similarity and semantic gravity disagree by design:

| Pair | cos(surface embed) | semantic gravity `F` | Correct direction |
|------|--------------------|----------------------|-------------------|
| (attract, repel) | high (co-occur in physics text) | `T = -1` → repulsive | repulsive |
| (gravity, anti-gravity) | high | `T = -1` | repulsive |
| (yes, no) | moderate | `T = -1` | repulsive |
| (gravity, library) | low | `T = 0` → neutral, but `M · M / d²` small | neutral |
| (gravity, momentum) | moderate | `T = +1`, `M · M / d²` moderate | attractive |

### §4.2 What this proves

A scalar-unsigned metric cannot distinguish rows 1-3 from row 5. Ternary-signed gravity does, by construction.

### §4.3 A note on learned attention

One could object that a learned attention head can approximate the sign. The response: (a) attention is opaque and non-falsifiable at the substrate level; (b) semantic gravity is explicit and editable by the curator; (c) per `feedback_attention_is_ternary_plus_contrastive.md`, K3D's attention *is* ternary-plus-contrastive and uses the same primitive.

---

## §5 Runtime Evidence (~0.75 page)

### §5.1 Implementation in the nine-chain swarm

At each tick of `trm_step_fused.ptx`, the nine parallel cognitive lanes evaluate `F(s_current, s_candidate)` for each candidate in the Galaxy neighbourhood. The top-`k` by `|F|` are promoted; sign determines whether the lane pursues or avoids the candidate.

### §5.2 Concrete run — math query

Worked example: `What is 2+3?` → Galaxy neighbourhood traversal shows attractive force from `S_ADDITION → S_DIGIT_2 → S_DIGIT_3 → S_INTEGER_5`; repulsive `T = -1` correctly prunes `S_SUBTRACTION` from the path.

### §5.3 Concrete run — ARC-AGI-3 live solve

Referenced from Paper A §5.2: the same mechanism selected visual-primitive candidates in `ls20-9607627b` Level 1.

### §5.4 Ablation — replacing T with scalar

Substitute `T ≡ +1` in the implementation; rerun the adversarial pair set from §4.1. Expected result: rows 1-3 misclassified. (Paper should include this ablation table.)

---

## §6 Discussion (~0.5 page)

### §6.1 Placement relative to Gärdenfors

Gärdenfors' conceptual-spaces framework gives the *space*; semantic gravity gives the *force law* that operates in it at runtime. The two are complementary, not competing.

### §6.2 Placement relative to embedding similarity

Embedding similarity is a useful *approximation to `d`* at ingestion time. It is not a substitute for `F`. §4's adversarial pairs establish the distinction.

### §6.3 Limitations

- `T` requires curatorial seeding for non-trivial pairs; learning `T` from corpora is future work.
- `M` weights α-δ are currently hand-tuned; principled calibration from Knowledgeverse statistics pending.
- Experiments are single-hardware and single-language-family in the current evaluation set.

### §6.4 Relationship to House vs Galaxy distinction

Per `HOUSE_VS_KNOWLEDGEVERSE_DISTINCTION.md`, semantic gravity operates *only* in the Galaxy (fluid, emergent). The House (intentional, curated) does not use it; the two coexist via bidirectional symlinks. This paper restricts claims to the Galaxy.

---

## §7 Conclusion (~0.25 page)

Three sentences:

1. Semantic gravity is a computable, ternary, language-agnostic force law over meaning-centric concept stars.
2. It is load-bearing (not decorative): ablation removes the ternary sign and adversarial pairs misclassify.
3. It bridges Gärdenfors' conceptual-spaces theory and K3D's GPU-native cognition substrate — the first full round-trip from a philosophical account of meaning to a runtime evaluating at frame cadence.

---

## Page Budget Check

| Section | Words | Pages (approx) |
|---------|-------|----------------|
| Abstract + authorship note | ~250 | 0.35 |
| §1 Introduction | 450 | 0.65 |
| §2 Background | 500 | 0.75 |
| §3 Formula and definitions | 850 | 1.25 |
| §4 Adversarial demonstration | 500 | 0.75 |
| §5 Runtime evidence | 500 | 0.75 |
| §6 Discussion | 325 | 0.5 |
| §7 Conclusion | 175 | 0.25 |
| References | — | ~0.5-0.75 |
| **Total** | **~3550 words + 2-3 tables + refs** | **~5.75 pages** |

Fits a typical 6-8 page preprint budget; trimmable if the venue enforces 6.

---

## Writing-phase todos

- [ ] Coordinate with Christoph Dorn on co-author order, approvals, and any edits to the authorship-note footnote.
- [ ] Populate empirical defaults for α, β, γ, δ in §3.6 from actual Knowledgeverse statistics (Codex experiment).
- [ ] Run the §5.4 ablation on the adversarial pair set and fill the table.
- [ ] Confirm ATTRIBUTIONS.md §4.4.1 language is verbatim-consistent with the §Authorship Note footnote.
- [ ] Decide whether to mention the term-coinage date (March 2026) in the authorship note or keep it in ATTRIBUTIONS only.
- [ ] Check citation formatting for Gärdenfors 2000, Collins & Loftus 1975, Fruchterman-Reingold 1991, Knuth TAOCP, BitNet-b1.58.

---

## What this skeleton is NOT

- Not a draft — prose targets, not submission wording.
- Not Christoph-approved — authorship note is drafted by Claude on Daniel's instruction; needs Christoph sign-off per Daniel's standing rule for co-author courtesy.
- Not empirically populated — §3.6 and §5 numerics pending Codex experimental runs.

---

**Location**: `TEMP/CLAUDE_PAPER_B_SKELETON_DRAFT_04.19.2026.md`
**Blocks**: Paper B draft (skeleton is prerequisite).
**Blocked by**: Christoph co-author confirmation, Codex empirical runs for §3.6 and §5.4.
**Parallel to**: [`CLAUDE_PAPER_A_SKELETON_DRAFT_04.19.2026.md`](CLAUDE_PAPER_A_SKELETON_DRAFT_04.19.2026.md).
