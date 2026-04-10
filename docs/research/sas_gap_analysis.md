| Gap | K3D Already Has | Delta (Missing) | PTX Kernel/Opcode |
|-----|-----------------|-----------------|-------------------|
| **A: Canonical Form** | `star_id=SHA-256(meaning_rpn|meaning_class|domain)`.<br>RPN is canonical for expression tree shape (commutative canonicalization is optional). | **Normalization pre-hash**: commutative operand sorting, rational/trig simplification to canonical form (e.g., `sin(-x)` → `-sin(x)`). | `OP_CANONICALIZE` (`0x229`) already in CAS registry. Delta: ensure it includes commutative normalization before star_id generation. |
| **B: Semantic Binding** | `reality_refs`, `grammar_refs`, `meta_refs` lists; bi‑directional symlinks; symbol `G` already appears in Math & Reality galaxies. | **Dynamic binding resolution**: given symbol `G` in Math Galaxy, auto‑resolve its Reality Galaxy entry via symlinks during semantic evaluation. | `OP_SEMANTIC_RESOLVE` (`0x22E`) – takes a star_id and target galaxy type, returns bound star_id(s). New in SAS range. |
| **C: Pattern‑Match‑Replace** | Grammar Galaxy stores rules as stars (`meaning_class="cas_rule"`, `pattern_rpn`, `behavior_rpn`); opcodes `OP_GRAMMAR_QUERY` (`0xE9`), `OP_PATTERN_MATCH` (`0x22C`), `OP_RULE_APPLY` (`0x22D`) planned. | **Rule‑priority & conflict‑resolution engine**: when multiple rules match, choose based on specificity, domain constraints, or user‑defined priority. | `OP_RULE_SELECT` (`0x22F`) – evaluates candidate rules, returns best match. New in SAS range. |

**Proposed new SAS opcodes (0x238–0x25F):**
1. `OP_SEMANTIC_RESOLVE` (`0x238`) – as above.
2. `OP_RULE_SELECT` (`0x239`) – as above.
3. `OP_CONTEXTUAL_REWRITE` (`0x23A`) – applies rule selected by `OP_RULE_SELECT` within a given domain context.
4. `OP_SEMANTIC_EQUIV` (`0x23B`) – checks if two stars are semantically equivalent (via canonical form and cross‑galaxy bindings).
5. `OP_GALAXY_FUSE` (`0x23C`) – merges two galaxies’ symlink networks for unified semantic resolution.