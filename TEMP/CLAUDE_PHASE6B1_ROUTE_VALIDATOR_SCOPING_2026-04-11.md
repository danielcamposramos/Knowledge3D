# CLAUDE → CODEX — Phase 6.B.1: Route Validator Scoping (legacy vs. symlink-dispatch) — 2026-04-11

## The situation

Full-boot integration of the 6.B crafter stars is blocked at [sovereign_hot_path.py:1405](../knowledge3d/knowledgeverse/sovereign_hot_path.py#L1405) by `sovereign_build_route_invalid: grammar_binary_op_infix:router_missing_executor_refs, grammar_binary_op_infix:router_missing_validator_refs, ...` for the five new grammar router stars.

**Do NOT fix this by adding `executor_refs` and `validator_refs` to the crafter grammar stars.** That validator is enforcing a legacy nine-chain specialist-swarm invariant (router → executor → validator triple) from the Python-orchestration era. The new meaning-centric stars per [MEANING_CENTRIC_STAR_SCHEMA_SPECIFICATION.md §2.3](../docs/vocabulary/MEANING_CENTRIC_STAR_SCHEMA_SPECIFICATION.md) use **symlink-as-dispatch**: Jarvis (Worker 8) follows a star's bidirectional symlinks to decide which specialist/program to run. There is no router→executor→validator triple in that model. Forcing the crafter stars to carry fake triple refs would be exactly the "Python orchestration band-aid" Daniel rejected.

The hook for the correct fix already exists. [sovereign_hot_path.py:1365](../knowledge3d/knowledgeverse/sovereign_hot_path.py#L1365) already reads a `sovereign_route_exempt` flag from the source row and copies it onto the translated star dict. The legacy validator at [sovereign_hot_path.py:1388-1405](../knowledge3d/knowledgeverse/sovereign_hot_path.py#L1388) just ignores it.

## The fix (three edits)

### 1. Scope the legacy validator

`_validate_route_link_coverage` in [sovereign_hot_path.py:1388](../knowledge3d/knowledgeverse/sovereign_hot_path.py#L1388):

```python
def _validate_route_link_coverage(self, stars: list[dict[str, Any]]) -> None:
    route_errors: list[str] = []
    for star in stars:
        if bool(star.get("sovereign_route_exempt")):
            continue
        role = str(star.get("selection_role") or "unknown")
        if role == "router":
            ...
```

Legacy specialist-route stars still enforce the triple. Meaning-centric crafter stars opt out.

### 2. Star Crafter marks its outputs exempt

Every star emitted by `knowledge3d/ingestion/star_crafter.py` sets `sovereign_route_exempt=True` on the source dict. This is not a loophole — it is the explicit declaration that the star participates in the symlink-dispatch routing model of `MEANING_CENTRIC_STAR_SCHEMA §2.3`, not the legacy router/executor/validator triple model.

Put this on the star-construction helper (or whatever the crafter's single emission path is called) so no crafter output can bypass it. Same discipline as the bidirectional `_link(...)` helper: one path, no shortcuts.

### 3. Symlink-closure invariant (the real guardrail)

The crafter stars are exempt from the legacy triple check, but they are NOT exempt from invariants. Add a new invariant that applies to meaning-centric (exempt) stars: **every symlink on an exempt star must resolve to an existing star in the same build.**

New helper alongside `_validate_route_link_coverage`:

```python
def _validate_symlink_closure(self, stars: list[dict[str, Any]]) -> None:
    star_ids = {str(s.get("id")) for s in stars if s.get("id")}
    closure_errors: list[str] = []
    for star in stars:
        if not bool(star.get("sovereign_route_exempt")):
            continue
        for field in ("taxonomy_refs", "component_refs", "composite_of",
                      "grammar_refs", "meta_refs", "visual_refs",
                      "audio_refs", "reality_refs"):
            for ref in list(star.get(field) or []):
                target_id = str(ref.get("star_id") if isinstance(ref, dict) else ref)
                if target_id and target_id not in star_ids:
                    closure_errors.append(f"{star.get('id')}:{field}:missing_target:{target_id}")
    if closure_errors:
        sample = ", ".join(closure_errors[:12])
        if len(closure_errors) > 12:
            sample += f", ... (+{len(closure_errors) - 12} more)"
        raise ValueError(f"sovereign_build_symlink_closure_invalid:{sample}")
```

Call it right after `_validate_route_link_coverage` in the same build step. Use whatever the canonical ref-field names are in `star_crafter.py` — this list is indicative, not authoritative. If the crafter stores refs under different keys, match those.

This is the meaning-centric equivalent of route closure: not "does this star reach a router/executor/validator triple," but "are this star's symlinks internally consistent." Bidirectional links created through `_link(...)` should always pass this trivially; the guardrail catches the case where a ref is written without its reverse.

## Why this is the right shape

- The exemption flag already existed in the translator — this is honoring a pre-wired capability, not inventing one.
- Legacy specialist stars still enforce their triple model (no regression on old paths).
- Meaning-centric stars enforce a different invariant (symlink closure) that matches how they actually route per schema §2.3.
- The crafter does not learn about the legacy validator at all. It only knows its own contract: emit a star, mark it exempt, register bidirectional symlinks.
- `sovereign_route_exempt` becomes the honest boundary flag between legacy specialist-swarm stars and meaning-centric crafter stars. Sleep-time can later walk this flag to decide what to consolidate vs. relocate to the Museum.

## What NOT to do

- Do not add `executor_refs` or `validator_refs` to any crafter star. Ever.
- Do not rename `grammar_binary_op_infix`'s `selection_role` away from `router` just to escape the check — its role is correct per §2.3 ("grammar stars route, not answer"). The problem was never the role; it was the validator applying the wrong invariant.
- Do not gate the exemption on a star_id prefix or a name regex. The only legitimate signal is the `sovereign_route_exempt` boolean written explicitly by the crafter at emission time.
- Do not widen the exemption to *all* stars without the symlink-closure invariant landing in the same change. Exemption without replacement invariant = no invariant.

## Validation

Add `tests/test_phase6b1_route_validator_scoping.py`:

1. **Legacy path unchanged** — synthesize a router star with no executor_refs, `sovereign_route_exempt=False`, assert boot raises `sovereign_build_route_invalid`.
2. **Exempt path accepted** — synthesize a router star with no executor_refs, `sovereign_route_exempt=True`, with one valid bidirectional symlink to a peer star, assert boot succeeds.
3. **Symlink closure enforced** — synthesize an exempt star with a symlink to a non-existent star_id, assert boot raises `sovereign_build_symlink_closure_invalid`.
4. **Full-boot integration** — boot the full Knowledgeverse with the crafter active, assert the 41,134-star table builds without route or closure errors.

Then re-run:
- `tests/test_phase6b_star_crafter.py` — must stay 7/7.
- Phase 1–5 regression batch — must stay green.
- `git diff --check` — clean.

## Then, and only then

Once full boot is green, run the two reprobes I proposed in the previous turn:

1. **6.A raw-record reprobe** on `math_operator_addition`, `concept_digit_two`, `rpn_program_addition`, `word_digit_two_en`, `grammar_binary_op_infix`. Expected:
   - operator star: `selection_role_id = ROLE_EXECUTOR`, `answer_eligible = 1`, `meta_rule_addr != 0`, `program_length > 0`
   - digit concept: `selection_role_id = ROLE_ANSWER`, `answer_eligible = 1`, `meta_rule_addr = 0`
   - rpn_program star: `meta_rule_addr` matches the operator's, `program_length` matches bytecode length
   - grammar star: `sovereign_route_exempt` observable via whatever 400-byte slot holds the flag
2. **Cosine routing probe** — embed `"plus"`, `"two"`, `"addition"`, `"3"` via the same sentence-transformer the crafter used, dump top-5 cosine hits against the live 41,134-star table. Assert the operator/digit meaning-stars appear in top-3.

Pass both and 6.C is greenlit.

## Handoff

- `plan_task` this spec to the ollama specialist first if the validator edit is more entangled than it looks from the lines above — there may be a second code site that re-derives route closure for a later consolidation step. If so, scope both sites under the same exemption flag.
- Flag any new `router`-role stars that come from the legacy `foundational_galaxy_builder` path that happen to carry `sovereign_route_exempt=True` — those would be accidental exemptions and must be investigated, not accepted.
- If the symlink-closure invariant catches real broken refs in the crafter output, that is *also* a win — fix the crafter, not the invariant.
