# RETE at the Opcode Level — Concrete Demonstration

**Date**: April 18, 2026
**Status**: Architecture reference — concrete illustration, not a new spec
**Purpose**: Show **exactly** how Forgy's RETE algorithm (1982) is realized in K3D at the RPN-opcode level, so future readers do not have to reconstruct the mapping from prose.

---

## Why This Document Exists

The RETE algorithm is referenced in `ATTRIBUTIONS.md`, in the Defeasible Logic integration, and in the Reasoning-Paradigm Block (`RPN_DOMAIN_OPCODE_REGISTRY.md §7`). What was missing was a concrete, end-to-end demonstration showing **which opcodes fire for which RETE construct**, using a specific rule traced through compilation and execution. This document fills that gap.

K3D does not embed a RETE library. It **compiles** RETE-structured inference to a small, deterministic block of RPN opcodes that live permanently in the Reasoning-Paradigm Block (`0xA0–0xF1`), with the three RETE-dedicated opcodes being:

| Opcode | Mnemonic | Role | Stack Effect |
|--------|----------|------|--------------|
| `0xE0` | `RETE_ALPHA_TEST` | Alpha-memory test (single-fact predicate) | `[fact, alpha_node] -> [match]` |
| `0xE1` | `RETE_BETA_JOIN`  | Beta-memory join (multi-fact pattern)   | `[left_token, right_token] -> [joined_token]` |
| `0xE2` | `AGENDA_INSERT`   | Conflict-set / agenda insertion         | `[activation] -> [agenda_handle]` |

All three are defined in [`RPN_DOMAIN_OPCODE_REGISTRY.md`](RPN_DOMAIN_OPCODE_REGISTRY.md) §7 lines 269–271. Their reservation authority is `TEMP/CLAUDE_REASONING_PARADIGMS_AND_N_SWARM_SPEC_2026-04-13.md §4`.

---

## RETE Background (One Paragraph)

RETE (Forgy 1982, *Artificial Intelligence 19(1)*) is a discrimination-network algorithm for matching patterns against a working memory of facts. Instead of re-evaluating every rule against every fact on every cycle, RETE compiles rules into a network of **alpha nodes** (per-fact predicate tests, cached in *alpha memories*) and **beta nodes** (multi-pattern joins over alpha memories, cached in *beta memories*). When a new fact arrives, only the affected nodes re-evaluate. Matches become **activations**, which are placed on an **agenda** (the conflict set) and fire according to a conflict-resolution strategy.

K3D reuses the *structure* of RETE — alpha test, beta join, agenda — but replaces its object-oriented runtime with three RPN opcodes that execute on GPU inside the Reasoning-Paradigm Block. The Galaxy Universe stores the discrimination network as metadata on Grammar Galaxy rules; the RPN opcodes walk it.

---

## Concrete Example — A Grammar Galaxy Rule

Consider a single defeasible Grammar Galaxy rule, written as Datalog-style for readability:

```
R1:  IF  token(t, kind=DIGIT)
     AND next(t, t')
     AND token(t', kind=DIGIT)
     THEN emit(number_candidate, span=[t, t'])
```

This rule has:
- **Two alpha tests** — both on `token(_, kind=DIGIT)`
- **One beta join** — linking two digit-tokens via `next(t, t')`
- **One activation** — emitting `number_candidate` when the full pattern matches

---

## Step 1 — Rule Compilation (Ingestion-Path, Not Hot-Path)

At ingestion time (flexible path), the Grammar Galaxy entry for `R1` is stored as structured metadata:

```jsonc
{
  "rule_id": "R1_NUMBER_CANDIDATE",
  "rule_strength": "defeasible",        // trit
  "alpha_tests": [
    {"alpha_id": "A1", "predicate": "kind == DIGIT", "slot": "token.kind"},
    {"alpha_id": "A2", "predicate": "kind == DIGIT", "slot": "token.kind"}
  ],
  "beta_joins": [
    {"beta_id": "B1", "left_alpha": "A1", "right_alpha": "A2",
     "join_predicate": "next(A1.t, A2.t)"}
  ],
  "agenda_entry": {
    "agenda_id": "G1_EMIT_NUMBER",
    "priority": "+d",                   // defeasible support
    "action": "emit(number_candidate, span=[A1.t, A2.t])"
  }
}
```

This JSON is pre-computed once and stored as a Grammar Galaxy node. It is **data**, not code — the hot path never parses it.

---

## Step 2 — RPN Compilation (Ingestion-Path)

The Grammar Galaxy entry is then compiled to a fixed RPN program that the Reasoning-Paradigm Block will execute. The compiled program is a linear sequence of opcodes using the three RETE opcodes plus the standard stack manipulation opcodes from the Base Tier:

```
; Rule R1_NUMBER_CANDIDATE
; Inputs on stack (from the swarm's current working set):
;   s1 = fact_handle_t   (the candidate first-token fact)
;   s2 = fact_handle_t2  (the candidate second-token fact)

  PUSH_CONST   alpha_node_A1         ; 0x01
  ROT          s1 -> top             ; bring fact to TOS
  RETE_ALPHA_TEST                    ; 0xE0  → trit match_A1
  TBRANCH_FAIL end_rule              ; short-circuit on -1

  PUSH_CONST   alpha_node_A2
  ROT          s2 -> top
  RETE_ALPHA_TEST                    ; 0xE0  → trit match_A2
  TBRANCH_FAIL end_rule

  ; Beta join: pair left & right tokens under the `next` predicate
  PUSH_CONST   beta_node_B1
  PACK_TOKENS  (s1, s2) -> left, right
  RETE_BETA_JOIN                     ; 0xE1  → joined_token (or 0 on fail)
  TBRANCH_ZERO end_rule

  ; All alpha + beta matched → produce an agenda activation
  PUSH_CONST   agenda_G1_EMIT_NUMBER
  AGENDA_INSERT                      ; 0xE2  → agenda_handle

end_rule:
  HALT_SET  local_halt               ; 0xF0 — feed swarm halt gate
```

The full program is stored as a procedural-memory star in Grammar Galaxy. The three RETE opcodes appear in exactly their canonical RETE roles:
- `0xE0` twice — once per alpha test
- `0xE1` once — for the beta join
- `0xE2` once — to insert the activation into the agenda

---

## Step 3 — Hot-Path Execution (Sovereign)

At inference time, the swarm scans the current working memory (a window of facts in VRAM) and invokes the compiled RPN program per candidate token-pair. Execution is PTX-native:

### Alpha Test (0xE0) — What It Actually Does

- **Semantics**: `[fact, alpha_node] -> [match]` where `match ∈ {+1, 0, -1}` (ternary match / unknown / mismatch).
- **Galaxy lookup**: The `alpha_node` operand is a star ID in Grammar Galaxy pointing to the predicate metadata. The opcode reads the single slot (`token.kind`) from the fact, compares against the predicate (`== DIGIT`), and pushes the trit.
- **Cache**: Alpha memory is represented by a Grammar Galaxy sub-star per alpha node — newly matching facts are symlinked to it during swarm consolidation. No per-rule re-evaluation is needed for already-matched facts.
- **Ternary advantage**: The `unknown` trit (`0`) propagates naturally through the rest of the pipeline, so missing data doesn't force a hard reject. This is a K3D-native extension to classical RETE (classical RETE is boolean).

### Beta Join (0xE1) — What It Actually Does

- **Semantics**: `[left_token, right_token] -> [joined_token]` — the two input tokens come from successful upstream alpha tests (or prior beta joins). The opcode evaluates the **join predicate** (`next(A1.t, A2.t)` in our example) and produces a joined token carrying the conjunction of the parents' trits via `TMUL`.
- **Galaxy lookup**: The `beta_node` operand points to the beta-memory star, which holds the join predicate as a small RPN sub-program (e.g., a `next(t1, t2)` check reduced to stack operations). This sub-program runs *inline* in the beta opcode — no recursion, bounded instruction count.
- **Cache**: Beta memory is a symlink set of successful `(left, right)` pairs on the beta-node star. The join returns `0` if either parent is unknown, `-1` if either parent is a mismatch, and `+1` if both match and the join predicate succeeds.

### Agenda Insert (0xE2) — What It Actually Does

- **Semantics**: `[activation] -> [agenda_handle]` — the activation payload is built from upstream beta-join outputs plus the rule's `agenda_entry` metadata.
- **Conflict-set handling**: The opcode inserts the activation into a per-lane agenda buffer ordered by **priority trit** (e.g., strict `+D` over defeasible `+d`) and by rule `trust_weight`. This is where defeasible logic joins RETE: superiority relations re-order the agenda before firing.
- **Halting integration**: The resulting `agenda_handle` is later consumed by `HALT_SET` / `HALT_SYNC` (`0xF0` / `0xF1`) at the end of the tick — a fired activation can push the halt trit to `+1` (done), `0` (continue), or `-1` (rejected, reset lane).

---

## Step 4 — Swarm × RETE Integration

Each swarm lane runs its own RPN program over its assigned candidate set:

```
Lane 0:  [fact_42, fact_43]   → R1 program → agenda_handle_G1_h0
Lane 1:  [fact_42, fact_44]   → R1 program → agenda_handle_G1_h1
Lane 2:  [fact_55, fact_56]   → R1 program → fail (alpha mismatch)
...
Lane 8:  [fact_99, fact_100]  → R1 program → agenda_handle_G1_h8
```

At the end of the tick, `HALT_SYNC` (`0xF1`) reduces all lane agendas. The Defeasible Logic Resolver (`gre_defeasible_resolver.cu`, which sits between the Nine-Chain Swarm and the Halting Gate per `ATTRIBUTIONS.md §4.4`) applies superiority relations, producing a verdict trit per candidate. Winning activations fire; losing ones are defeated. No Python orchestrates any of this.

---

## Mapping RETE Concepts to K3D Locations

| RETE Concept (Forgy 1982) | K3D Realization |
|---------------------------|-----------------|
| Alpha node                | Grammar Galaxy sub-star (predicate metadata) |
| Alpha memory              | Symlink set on the alpha-node star |
| Alpha test                | Opcode `0xE0 RETE_ALPHA_TEST` |
| Beta node                 | Grammar Galaxy sub-star (join predicate as inline RPN) |
| Beta memory               | Symlink set of matched (left, right) pairs |
| Beta join                 | Opcode `0xE1 RETE_BETA_JOIN` |
| Activation                | Token pushed by `RETE_BETA_JOIN` |
| Agenda / conflict set     | Per-lane agenda buffer ordered by priority trit |
| Agenda insertion          | Opcode `0xE2 AGENDA_INSERT` |
| Conflict resolution       | Defeasible Logic Resolver (`gre_defeasible_resolver.cu`) |
| Firing / retracting       | Halting gate `0xF0 HALT_SET` + `0xF1 HALT_SYNC` |

---

## What K3D Adds to Classical RETE

1. **Ternary matching** — every test returns `{+1, 0, -1}` instead of boolean, so partial / unknown facts propagate instead of forcing early failure.
2. **Defeasible priority** — the agenda is ordered by trit-encoded rule strength (`rule_strength` trit) and can be re-ranked by superiority relations before firing.
3. **GPU-parallel evaluation** — alpha and beta evaluations run across all swarm lanes in parallel; there is no single-threaded match loop.
4. **Memory-palace embedding** — alpha and beta memories are stars in Grammar Galaxy, so they are inspectable by humans (dual-client contract) and consolidated during sleep-time rather than held only in transient process memory.
5. **Sovereign runtime** — no RETE library is imported. The three opcodes, plus the standard stack opcodes, are the entire runtime. (Classical RETE implementations ship as libraries in C, Java, or Python.)

---

## Why This Matters

Readers who know RETE can now trace, opcode by opcode, how a pattern-matching production rule lands on the GPU. No ambiguity, no hand-waving. The discrimination network is real, the memories are real, the agenda is real — they live in Grammar Galaxy as symlinked stars, and they are walked by three opcodes in `0xE0–0xE2`.

This makes the novelty claim precise: K3D does not "use RETE-inspired ideas"; it **compiles RETE** onto a small, auditable opcode surface under the sovereignty constraint. That compilation is the contribution.

---

## References

- Forgy, C. L. (1982). *Rete: A fast algorithm for the many pattern / many object pattern match problem.* Artificial Intelligence 19(1), 17–37.
- [`RPN_DOMAIN_OPCODE_REGISTRY.md §7`](RPN_DOMAIN_OPCODE_REGISTRY.md) — Reasoning-Paradigm Block (`0xA0–0xF1`)
- `TEMP/CLAUDE_REASONING_PARADIGMS_AND_N_SWARM_SPEC_2026-04-13.md §4` — reservation authority
- [`ATTRIBUTIONS.md §4.4`](../../ATTRIBUTIONS.md) — Defeasible Logic integration and Christoph Dorn's contribution
- `TEMP/CLAUDE_DEFEASIBLE_LOGIC_INTEGRATION_03.16.2026.md` — architecture spec for the resolver that consumes the agenda

---

**License**: CC-BY-4.0 (Documentation)
**Version**: 1.0 (2026-04-18)
