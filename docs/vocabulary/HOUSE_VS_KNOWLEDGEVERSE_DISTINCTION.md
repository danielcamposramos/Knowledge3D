# House vs Knowledgeverse — Distinction Clarifier

**Date**: April 18, 2026
**Purpose**: Document, with a concrete disagreement example, why House ordering and Knowledgeverse (Galaxy) ordering are **not the same thing** and must not be collapsed. Written in response to the standing feedback *"House = intentional librarian placement. Galaxy = fluid semantic gravity during reasoning."*
**Status**: Architecture reference — clarifier, not a new spec.

---

## TL;DR

The **House** is an *intentional* spatial arrangement — a librarian's placement of knowledge objects in physical rooms, shelves, and chapters. Placement is chosen by a human or by the system with explicit curatorial intent.

The **Knowledgeverse** (specifically the Galaxy Universe region) is a *fluid* semantic-gravity field — placement emerges from the meaning-mass / meaning-distance formula `F = T(s₁,s₂) · M(s₁) · M(s₂) / d²` between stars. Nothing is "placed"; everything settles where the field pulls it.

These two orderings **disagree**, and the disagreement is load-bearing: reasoning uses the fluid field, curation uses the intentional placement, and symlinks connect them. Collapsing the two would either make the House rigid (destroying reasoning flexibility) or make the Knowledgeverse arbitrary (destroying curatorial trust).

---

## The Two Substrates, Side by Side

| Aspect | House | Knowledgeverse (Galaxy Universe) |
|--------|-------|----------------------------------|
| **Persistence** | SSD (GLB + JSONL) | VRAM (working memory) |
| **Placement driver** | Intentional — librarian / curator decision | Emergent — semantic-gravity formula |
| **Primary metaphor** | Method of Loci / Memory Palace | Physics — meaning as mass, distance as proximity |
| **Ordering signal** | `(x, y, z)` coordinates in 3D space + shelf / room / chapter indices | `M(star)`, `T(s₁, s₂)`, `d(s₁, s₂)` — meaning, ternary operator, meaning-distance |
| **Updated how** | Human edits + sleep-time consolidation | Reasoning-time re-balancing, every tick |
| **Updated how often** | Slowly (intentional changes) | Continuously during cognition |
| **Client** | Humans primarily (avatar lives here), AI as visitor | AI primarily (Galaxy = brain), humans as inspectors |
| **Determinism** | Deterministic — same placement until edited | Dynamic — force field depends on current context |
| **Loss surface** | Curatorial error (misplaced book) | Meaning error (wrong meaning-mass / wrong ternary operator) |

The key is the **placement driver** row: *intentional* vs *emergent*.

---

## Concrete Disagreement Example

Suppose the Knowledgeverse contains the following five stars (each is a meaning-centric concept with multilingual symlinks):

| Star ID | Concept | Domain |
|---------|---------|--------|
| `S_GRAVITY` | "gravity" (the physical force) | Physics |
| `S_MOMENTUM` | "momentum" | Physics |
| `S_ATTRACTION` | "attraction" (general semantic sense) | Affect / general language |
| `S_LIBRARY` | "library" (a place of books) | Architecture / institutions |
| `S_BOOK` | "book" (the concept, not an instance) | Physical object / artifact |

### Knowledgeverse (Galaxy) Ordering

The semantic-gravity field computes, for each pair, `F(sᵢ, sⱼ)`. Applied to these five stars, the resulting adjacency (strongest pulls first):

```
S_GRAVITY  —  S_MOMENTUM       (both physics, high meaning-mass shared context, small d)
S_GRAVITY  —  S_ATTRACTION     (semantic-force relation; T operator stays supportive)
S_LIBRARY  —  S_BOOK           (institutional-object relation)
S_ATTRACTION — S_BOOK          (WEAK — "attractive book" is a valid phrase but
                                distant semantic-mass)
```

Notice that `S_GRAVITY` does **not** cluster near `S_LIBRARY` or `S_BOOK` under the semantic-gravity field, because their meaning-distance is large and the ternary operator `T(gravity, library)` yields near-zero support.

### House Ordering

Now suppose the House curator (Daniel, or the Reality Enabler at construction time) has deliberately placed:

- `S_GRAVITY` and `S_ATTRACTION` as **books on a shelf in the Physics Room** (3D coordinates inside the Physics Room GLB)
- `S_LIBRARY` as the **name of the room itself** (the Library Room) — a structural label on the room entity
- `S_BOOK` as the **3D asset template** used for every book object in the House — so it lives as a Form-layer primitive, not a shelved concept
- `S_MOMENTUM` as a **book in a different room** (the Advanced Physics Room), behind a door

The House ordering, read by walking from the front door:

```
Living Room  →  Library Room  →  Physics Room  →  (door)  →  Advanced Physics Room
                   ^               ^                            ^
              S_LIBRARY         S_GRAVITY                  S_MOMENTUM
                                S_ATTRACTION
                                (shelved together by curator intent)
S_BOOK  →  every book-shaped 3D object uses this Form template
```

### Where They Disagree

1. **`S_GRAVITY` and `S_MOMENTUM`**
   - Galaxy: **strongly adjacent** (same domain, high pull).
   - House: **in different rooms, separated by a door** (curator put advanced-physics materials behind a door to signal "graduate-level content").

2. **`S_GRAVITY` and `S_ATTRACTION`**
   - Galaxy: **moderately adjacent** (the force relation pulls them together in the gravity field).
   - House: **shelved right next to each other** (curator explicitly placed them together to teach the analogy).

3. **`S_LIBRARY` and `S_BOOK`**
   - Galaxy: **adjacent** (institutional-object relation; books live in libraries).
   - House: **not in an adjacency relation at all** — one is a Room label, the other is a Form-level 3D asset template used everywhere. They're not even comparable positionally.

4. **`S_ATTRACTION` and `S_BOOK`**
   - Galaxy: **weak** (the phrase "attractive book" is valid but carries low meaning-mass at the pair level).
   - House: **no relationship** — one is a shelved physics concept, the other is a Form template.

### Why the Disagreement Matters

If we **used the House ordering for reasoning**, the system would conclude that `S_GRAVITY` and `S_MOMENTUM` are unrelated (they're behind a door in different rooms). That would be wrong for any physics inference.

If we **used the Galaxy ordering for curation**, the House would constantly reshuffle itself every time the AI reasoned about something new — books would fly between shelves whenever semantic gravity shifted. A human walking into the Library Room could not trust that tomorrow it would still look like a library.

**Both orderings are correct at what they do**. Using either for the other's job breaks the system.

---

## How They Connect — Symlinks

The two orderings stay coherent through **symlinks**, which are K3D's canonical reference form (always bidirectional per `feedback_bidirectional_symlinks_norm.md`).

- The **House** stores a 3D asset at coordinates `(x, y, z)` in the Physics Room. The asset carries a symlink `→ S_GRAVITY` (the meaning-centric star in the Knowledgeverse).
- The **Knowledgeverse** stores `S_GRAVITY` with meaning-mass, multilingual surface forms, RPN program, and a bidirectional symlink `← house://physics_room/shelf_2/slot_4`.

When a human walks to that shelf, they see the book (House view). When the AI reasons about physics, the semantic-gravity field over meaning-centric stars does the thinking (Galaxy view). The two views share the same canonical star; they just traverse it differently.

This is the **dual-client contract** in action: one canonical meaning, two consistent surface views, neither view determining the other's ordering.

---

## The Librarian Metaphor (House) vs the Physics Metaphor (Galaxy)

**House — The Librarian**

A good librarian places books with *intent*: `Physics` books go in the Physics section, arranged by sub-field, within that by era, within that by author. This placement is a **curatorial act**. It encodes the librarian's pedagogical choices, the institution's discipline boundaries, and social conventions about how knowledge is organized. It is **slow**, **deliberate**, and **legible** to humans. The House is this kind of space.

**Galaxy — The Physical Field**

A gravitational field doesn't "choose" anything. Bodies of mass pull each other by a physical law, distances emerge from initial conditions, clusters form where the field is densest. The field is **fast**, **continuous**, and **ground-truth consistent** — you cannot "overrule" a gravitational field by pointing at where a planet *ought* to be. The Knowledgeverse's Galaxy region operates under `F = T·M·M/d²` the same way.

Curators cannot dictate where the semantic-gravity field places things (it follows meaning-mass and ternary relations), and the semantic-gravity field cannot override where the curator placed a book on a physical shelf (placement is intentional data). The two respect each other's authority within their own domain.

---

## When to Use Which

| Question | Use the House | Use the Knowledgeverse (Galaxy) |
|----------|---------------|----------------------------------|
| "Where is this book physically?" | ✅ | — |
| "What domain does this concept belong to as a curator decided?" | ✅ | — |
| "What stars are semantically nearest to this concept right now?" | — | ✅ |
| "Which neighborhood should the swarm consider for this query?" | — | ✅ |
| "What does the human avatar see when they walk into the Library Room?" | ✅ | — |
| "What does the AI attend to when reasoning about gravity?" | — | ✅ |
| "Where should we render a book's 3D asset?" | ✅ | — |
| "Which specialist adapter should activate for this query?" | — | ✅ |

When a subsystem claims to answer one of the left-column questions by consulting the Galaxy, or one of the right-column questions by consulting the House, it is **mis-using the substrate**. Flag and fix.

---

## Common Mistakes (and How to Avoid Them)

1. **"Let's rank books on the shelf by semantic gravity."**
   - *No.* Shelf order is curatorial. The Galaxy's gravity re-orders every tick — the human would find a chaotic shelf.
   - *Do instead:* Symlink each shelved book to its meaning-star. The AI uses the Galaxy; the human uses the shelf.

2. **"Let's use 3D distance in the House as the query neighborhood for reasoning."**
   - *No.* Two concepts in adjacent rooms may be semantically far; two concepts on different floors may be semantically adjacent. House distance is **not** meaning distance.
   - *Do instead:* Use semantic-gravity neighborhoods in the Galaxy for reasoning. Use House geometry only for embodied navigation (avatar moving through rooms).

3. **"The same concept can't be in two places in the House."**
   - *Wrong.* A concept **can** appear at multiple House locations via symlinks (a reference book on the Physics shelf *and* in the Math Room). There's still one canonical star in the Galaxy — the House is just showing it at both places with intent.
   - *Do instead:* Accept multi-site symlinked placements as normal; the star is one, the placements are many, and both are legitimate.

4. **"Sleep-time consolidation should rewrite the House by Galaxy gravity."**
   - *No.* Sleep-time consolidation can **suggest** House rearrangements (e.g., "the Physics Room keeps being queried with the Math Room — consider a doorway"), but the curator (Daniel or the Reality Enabler's rules) must approve. The House's intentional-placement property is what makes it a trustworthy Memory Palace.
   - *Do instead:* Sleep-time crafts/prunes Galaxy specialist regions freely; House edits are proposed, not auto-applied.

---

## References

- [`KNOWLEDGEVERSE_SPECIFICATION.md`](KNOWLEDGEVERSE_SPECIFICATION.md) — the 7-region VRAM substrate
- [`KNOWLEDGEVERSE_TERM_DEFINITION.md`](KNOWLEDGEVERSE_TERM_DEFINITION.md) — canonical definition
- [`KNOWLEDGEVERSE_TERM_ORIGIN_PROOF.md`](KNOWLEDGEVERSE_TERM_ORIGIN_PROOF.md) — prior-art research
- [`THREE_BRAIN_SYSTEM_SPECIFICATION.md`](THREE_BRAIN_SYSTEM_SPECIFICATION.md) — Cranium + Galaxy + House
- [`MEMORY_TABLET_SPECIFICATION.md`](MEMORY_TABLET_SPECIFICATION.md) — how the interface tablet sits in the House
- `feedback_house_vs_galaxy_organization.md` — the standing feedback this document clarifies
- `feedback_book_is_galaxy_not_star.md` — related clarification on books as Galaxies, not stars
- `feedback_bidirectional_symlinks_norm.md` — why the connection is always bidirectional
- `ATTRIBUTIONS.md §4.4.1` — semantic gravity formula and provenance

---

**License**: CC-BY-4.0 (Documentation)
**Version**: 1.0 (2026-04-18)
