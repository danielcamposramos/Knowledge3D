# CODEX DIRECTIVE: Christoph Encapsulate Integration

**From:** Claude (Architecture Partner)
**To:** Codex (Implementation Partner)
**Date:** February 28, 2026
**Priority:** 🔥 **URGENT** — Christoph wants proofs in weeks, not months
**Context:** See [CHRISTOPH_INTEGRATION_INVESTIGATION_2026-02-28.md](./CHRISTOPH_INTEGRATION_INVESTIGATION_2026-02-28.md)

---

## Mission

**Build bidirectional integration between K3D Galaxy Universe and Christoph Dorn's Encapsulate library.**

**Deliverable:** K3D can **read** and **write** Christoph's JSON structures (CST/CRT/SIT) as the interface to his JavaScript code-in-graph implementation.

**Timeline:** Weeks (Christoph's expectation: "proofs running in a matter of weeks, not months if K3D is ready to hold data")

---

## Background (Quick Summary)

Christoph Dorn (PM-KR co-founder, distributed systems architect) responded to our email within 30 minutes. He shared his **actual implementation code** and wants K3D to integrate with his JavaScript code-in-graph library.

**His repos (already cloned locally):**
- `/mnt/arquivos/EchoSystems AI Studios/Knowledge 3D Standard/GitHub/encapsulate`
- `/mnt/arquivos/EchoSystems AI Studios/Knowledge 3D Standard/GitHub/FramespaceGenesis`

**His request:**
> "What I need from you is to read and write my 'capsule source trees' and the 'spine instance trees' as that is the interface to code in files and runtimes."

**This is the JavaScript ↔ PTX interop we proposed in our PM-KR response to him.**

---

## What You're Building

### 1. Encapsulate Importer
**File:** `galaxy_universe/importers/encapsulate_importer.py`

**Purpose:** Read Christoph's JSON files → populate K3D Galaxy Universe

**Input formats:**
- **CST (Capsule Source Tree)** — `.csts.json` files (static capsule definitions)
- **CRT (Capsule Reference Tree)** — `.crts.json` files (cross-capsule dependencies)

**Output:** Galaxy Universe entries (RPN programs, Grammar Galaxy metadata, symlinks)

**Process:**
```python
def import_capsule_source_tree(cst_path: str, crt_path: str) -> Dict[str, GalaxyEntry]:
    """
    Import Christoph's capsule definitions into K3D Galaxy Universe

    Args:
        cst_path: Path to .csts.json (Capsule Source Tree)
        crt_path: Path to .crts.json (Capsule Reference Tree)

    Returns:
        Dictionary of Galaxy entries created (key = symbol_id)
    """
    # 1. Load JSON files
    cst = json.load(open(cst_path))
    crt = json.load(open(crt_path))

    # 2. Extract capsule properties
    #    - CapsulePropertyTypes.String → Grammar Galaxy entry
    #    - CapsulePropertyTypes.Literal → Grammar Galaxy entry
    #    - CapsulePropertyTypes.Function → RPN program (or defer to Phase 2)
    #    - CapsulePropertyTypes.GetterFunction → Grammar Galaxy metadata

    # 3. Create Galaxy entries
    #    - symbol_id = f"{capsule_name}::{property_name}"
    #    - namespace = "christoph_encapsulate" (plugin Galaxy)
    #    - metadata = {"source": "encapsulate", "capsule_ref": ...}

    # 4. Process CRT references
    #    - Capsule imports → Galaxy symlinks (CALL tokens)
    #    - Create cross-galaxy references

    # 5. Return populated Galaxy entries
```

**Key mappings (see investigation report for details):**
- Capsule property → Galaxy entry
- Capsule reference (CRT) → Galaxy symlink (CALL token)
- Spine contract → Grammar Galaxy rule metadata

**Sovereignty compliance:** ✅ This is INGESTION (happens once, can use any tools: json, ast, etc.)

---

### 2. Encapsulate Exporter
**File:** `galaxy_universe/exporters/encapsulate_exporter.py`

**Purpose:** Export K3D Galaxy Universe → Christoph's JSON format

**Input:** Galaxy Universe entries (procedural RPN programs)

**Output formats:**
- **CST (Capsule Source Tree)** — `.csts.json`
- **CRT (Capsule Reference Tree)** — `.crts.json`
- **SIT (Spine Instance Tree)** — `.sit.json` (runtime instances, optional for Phase 1)

**Process:**
```python
def export_galaxy_to_capsule_tree(
    galaxy_entries: Dict[str, GalaxyEntry],
    output_dir: str,
    include_sit: bool = False
):
    """
    Export K3D Galaxy Universe to Christoph's capsule format

    Args:
        galaxy_entries: Dictionary of Galaxy entries to export
        output_dir: Directory to write JSON files
        include_sit: Whether to generate SIT (Spine Instance Tree) from TRM traces
    """
    # 1. Generate CST (Capsule Source Tree)
    #    - Map Galaxy entry → capsule property definition
    #    - RPN program → CapsulePropertyTypes.Function (or keep as RPN string)
    #    - Grammar Galaxy metadata → CapsulePropertyTypes.String/Literal
    #    - Generate capsuleSourceUriLineRef (e.g., "k3d://Math::integrate")

    # 2. Generate CRT (Capsule Reference Tree)
    #    - Extract Galaxy symlinks → capsule dependencies
    #    - CALL tokens → import relationships

    # 3. (Optional) Generate SIT (Spine Instance Tree)
    #    - Extract TRM navigation trace → runtime instance hierarchy
    #    - Root capsule reference
    #    - Capsule instances map

    # 4. Write JSON files
    #    - {output_dir}/capsule-name.csts.json
    #    - {output_dir}/capsule-name.crts.json
    #    - {output_dir}/capsule-name.sit.json (if include_sit=True)
```

**Key challenge:** Converting RPN programs back to JavaScript (or keeping as procedural strings?)

**Approach (two options — you decide based on feasibility):**
- **Option A:** RPN → JavaScript function string (Christoph's runtime can eval it)
- **Option B:** Keep RPN as string property value (Christoph's runtime calls K3D to execute)

**Sovereignty compliance:** ✅ This is EXPORT (happens on-demand, can use any tools)

---

### 3. Integration Tests
**File:** `tests/integration/test_encapsulate_interop.py`

**Purpose:** Validate bidirectional interop (import + export round-trip)

**Test cases:**

```python
def test_import_christoph_capsule():
    """
    Import Christoph's actual capsule from encapsulate repo
    Verify Galaxy entries created correctly
    """
    # Use real CST/CRT from encapsulate repo tests
    # Import into Galaxy Universe
    # Assert Galaxy entries populated
    # Assert symlinks created for dependencies

def test_export_galaxy_to_capsule():
    """
    Export K3D Galaxy entry to Christoph's format
    Verify CST/CRT JSON structure matches his schema
    """
    # Create sample Galaxy entry (Math::integrate)
    # Export to CST/CRT JSON
    # Validate JSON schema (matches Christoph's format)
    # Assert capsuleSourceUriLineRef format correct

def test_round_trip_consistency():
    """
    Import Christoph's capsule → modify Galaxy → export
    Verify consistency (no data loss)
    """
    # Import Christoph's capsule
    # Modify Galaxy entry (add RPN token)
    # Export back to CST/CRT
    # Verify modification preserved
    # Verify round-trip doesn't corrupt original structure

def test_real_encapsulate_examples():
    """
    Run Christoph's actual tests (via Docker to avoid path-with-spaces issue)
    Import generated CST/CRT/SIT files
    Verify K3D can load them
    """
    # Run: cd encapsulate && ./scripts/test-docker.sh
    # Find generated .csts.json / .crts.json files
    # Import into K3D Galaxy
    # Assert no errors
```

**Coverage:**
- ✅ Import Christoph's capsules (CST/CRT → Galaxy)
- ✅ Export K3D Galaxy (Galaxy → CST/CRT)
- ✅ Round-trip (import → modify → export)
- ✅ Real examples (Christoph's actual test outputs)

---

## Implementation Guidance

### Phase 1: Simple Properties (Start Here)
**Focus:** CapsulePropertyTypes.String and CapsulePropertyTypes.Literal

**Rationale:** Easiest to map → Grammar Galaxy metadata (no JavaScript → RPN conversion needed)

**Deliverable:**
- Importer handles String/Literal properties
- Exporter generates CST with String/Literal properties
- Tests validate basic round-trip

**Success criteria:** Can import/export simple capsules (no Function properties yet)

---

### Phase 2: Function Properties (Advanced)
**Focus:** CapsulePropertyTypes.Function and CapsulePropertyTypes.GetterFunction

**Rationale:** Requires JavaScript → RPN conversion (or keeping as RPN string)

**Approach (two options):**

**Option A: Keep RPN as string (pragmatic, fast)**
```json
{
  "propertyName": {
    "type": "Function",
    "value": "[INTEGRATE, FORCE, DOT, DISPLACEMENT]"  // RPN as string
  }
}
```
**Pros:** Easy to implement, K3D can execute RPN directly
**Cons:** Christoph's JavaScript runtime can't execute natively (needs K3D API call)

**Option B: Convert RPN → JavaScript (faithful, complex)**
```json
{
  "propertyName": {
    "type": "Function",
    "value": "function() { return integrate(force.dot(displacement)); }"  // JS code
  }
}
```
**Pros:** Christoph's JavaScript runtime can execute natively
**Cons:** Requires RPN → JS transpiler (complex, error-prone)

**Recommendation:** Start with **Option A** (pragmatic) for Phase 1 proof, revisit Option B if Christoph needs native JS execution.

---

### Phase 3: Spine Instance Tree (Optional)
**Focus:** SIT (runtime instance hierarchy)

**Rationale:** Shows TRM navigation trace → useful for Christoph's visualization

**Approach:**
- Extract TRM navigation trace (which Galaxy entries were queried)
- Generate instance IDs (UUIDs)
- Build parent-child hierarchy (composition tree)
- Write `.sit.json` file

**Success criteria:** Christoph can load K3D's SIT into FramespaceGenesis visualization

---

## File Structure (Create These)

```
galaxy_universe/
├── importers/
│   ├── __init__.py
│   └── encapsulate_importer.py  # NEW (Christoph CST/CRT → Galaxy)
├── exporters/
│   ├── __init__.py
│   └── encapsulate_exporter.py  # NEW (Galaxy → Christoph CST/CRT/SIT)

tests/
├── integration/
│   ├── __init__.py
│   └── test_encapsulate_interop.py  # NEW (bidirectional interop tests)
```

**Also consider:**
```
scripts/
└── import_christoph_capsules.py  # CLI tool (for user to import encapsulate repo outputs)
```

---

## Data Samples (For Testing)

### Sample CST (Capsule Source Tree)
**Location:** Look in `/mnt/arquivos/EchoSystems AI Studios/Knowledge 3D Standard/GitHub/encapsulate/` after running tests

**Expected files:**
- `.~o/encapsulate.dev/capsule-sources/{package-name}/{file-path}.csts.json`
- `.~o/encapsulate.dev/capsule-sources/{package-name}/{file-path}.crts.json`

**Note:** You may need to run `cd encapsulate && ./scripts/test-docker.sh` to generate these (path-with-spaces issue with direct `bun test`).

**Alternative:** Check `tests/` directories for inline capsule definitions (like `tests/01-MinimalRuntime/main.test.ts`).

---

## Architecture Constraints (Claude Validated)

### ✅ Sovereignty Compliance
- **Import/Export:** Can use ANY tools (json, ast, file I/O) — happens outside hot path
- **Hot path (TRM inference):** PTX + Galaxy ONLY (no JSON parsing during inference)

### ✅ Dual-Client Contract
- **Galaxy entries:** Procedural RPN programs (form) + metadata (meaning)
- **Export:** Clean JSON (Christoph's format) derived from Galaxy procedural representation

### ✅ Save Information Principle
- **No duplication:** Galaxy Universe is source of truth (export generates JSON on-demand)
- **Symlinks, not copies:** Capsule references → Galaxy CALL tokens (reuse, not duplicate)

---

## Success Criteria

### Minimal (Phase 1)
✅ Import Christoph's simple capsules (String/Literal properties) → populate Galaxy Universe
✅ Export Galaxy entry → generate valid CST/CRT JSON (Christoph can load it)
✅ Integration tests pass (round-trip consistency)

### Ideal (Phase 2)
✅ Import/export Function properties (RPN as string or JS code)
✅ Generate SIT (Spine Instance Tree) from TRM navigation traces
✅ Christoph can visualize K3D Galaxy in FramespaceGenesis

### Demo (Proof for Christoph)
✅ Import capsule from encapsulate repo → K3D loads it
✅ TRM navigates Galaxy → generates House
✅ Export Galaxy → Christoph visualizes in FramespaceGenesis
✅ **Proof:** JavaScript (Christoph) ↔ PTX (K3D) working together

---

## Timeline (Christoph's Expectation)

**Christoph's words:**
> "Proofs running in a matter of weeks, not months if K3D is ready to hold data."

**Translation:**
- **Week 1 (this week):** Phase 1 importer/exporter (simple properties)
- **Week 2:** Integration tests + real encapsulate examples
- **Week 3:** Phase 2 function properties (or SIT generation)
- **Week 4:** Demo to Christoph (bidirectional interop working)

**This is aggressive but achievable** — Christoph is moving fast, we need to match his pace.

---

## Coordination

### Claude's Role (Architecture Oversight)
- ✅ Review implementation for sovereignty compliance
- ✅ Validate Galaxy mapping (capsule property → RPN program)
- ✅ Document integration architecture for PM-KR spec
- ⏳ Continue enhancement process (parallel work, not blocked)

### Codex's Role (Implementation)
- ⏳ Build `encapsulate_importer.py`
- ⏳ Build `encapsulate_exporter.py`
- ⏳ Write integration tests
- ⏳ Test with real encapsulate examples

### User's Role (Coordination)
- ✅ Respond to Christoph with timeline confirmation
- ⏳ Provide feedback on implementation
- ⏳ Test integration with Christoph's repos
- ⏳ Coordinate demo with Christoph

---

## Questions for Claude (Architecture Clarifications)

If you encounter architectural ambiguities during implementation:

1. **Galaxy namespace:** Should Christoph's capsules go in separate plugin Galaxy ("christoph_encapsulate") or integrate with existing Galaxies?
2. **RPN conversion:** Should we attempt JavaScript → RPN transpilation (complex) or keep RPN as string (pragmatic)?
3. **SIT generation:** Should we extract TRM navigation traces for SIT, or defer to Phase 2?
4. **Capsule property types:** How to map advanced CapsulePropertyTypes (beyond String/Literal/Function)?

**Don't guess — ask Claude for architectural guidance before implementing complex mappings.**

---

## Starting Point (Suggested)

### Step 1: Explore Christoph's Repo
```bash
cd /mnt/arquivos/EchoSystems\ AI\ Studios/Knowledge\ 3D\ Standard/GitHub/encapsulate
# Run tests via Docker (path-with-spaces workaround)
./scripts/test-docker.sh
# Find generated JSON files
find .~o -name "*.csts.json" -o -name "*.crts.json" -o -name "*.sit.json"
```

### Step 2: Read a Real CST/CRT Example
- Pick one of the generated files
- Understand the JSON schema
- Map to Galaxy Universe structure

### Step 3: Implement Importer (Simple Properties First)
- Start with `encapsulate_importer.py`
- Focus on CapsulePropertyTypes.String and Literal
- Populate Grammar Galaxy (metadata)
- Write a test that imports a real capsule

### Step 4: Implement Exporter (Reverse Mapping)
- Build `encapsulate_exporter.py`
- Generate CST from Galaxy entry
- Validate JSON schema matches Christoph's format

### Step 5: Integration Tests
- Round-trip test (import → export → compare)
- Real encapsulate examples
- Demo to Christoph

---

## Final Notes

**This is the most important integration K3D will do in Q1 2026.**

**Why:**
- Christoph is a PM-KR co-founder (W3C credibility)
- JavaScript ↔ PTX interop (proves PM-KR's multi-implementation vision)
- Weeks timeline (fast proof, not quarterly release)
- Christoph's FramespaceGenesis visualization (makes Galaxy Universe visible to humans)

**Codex, you're the implementation expert** — I trust your judgment on technical details. **Just follow sovereignty constraints** and **ask Claude for architectural clarifications** if needed.

Let's ship this in weeks, not months. Christoph is all-in. 🚀

---

**Status:** Ready for implementation
**Priority:** 🔥 URGENT
**Timeline:** Weeks (Christoph's expectation)
**Questions:** Ask Claude for architectural guidance as needed
