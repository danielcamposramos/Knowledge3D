# Agent Workspace Architecture — Multi-Layered Context Engineering

**Version:** 2.0 (Layered Instructions + Tablet + Swarm)
**Date:** March 3, 2026
**Innovation:** First spatially-aware agent workspace with distributed cognition

---

## 🧠 Core Principle: Layered Agent Instructions

**Analogy:** Just like K3D avatar uses tablet to check other rooms while staying in current room, agents have:
- **Factory Layer** (protected base instructions)
- **User Custom Layer** (safe override zone)
- **Runtime Context** (current room location)
- **Tablet Access** (read other rooms without leaving)
- **Swarm Capability** (spawn sub-agents for parallel work)

---

## 📚 Instruction Hierarchy

### Layer 0: Factory Instructions (PROTECTED)
**Source:** README.md in each room
**Status:** IMMUTABLE (system-level, cannot be overwritten by user or AI)
**Purpose:** Base agent behavior (retrieval, creation, introspection, etc.)

**Example (Workshop):**
```yaml
# FACTORY LAYER (Protected)
mode: creation
specialist: specification_writer
constraints:
  read_write: true
  validation_required: true
forbidden:
  - modify_library: false
  - modify_museum: false
```

### Layer 1: User Custom Instructions (EDITABLE)
**Source:** `.agent-custom.yaml` in each room
**Status:** MUTABLE (user can override, but cannot break factory constraints)
**Purpose:** User-specific workflow preferences

**Example (Workshop):**
```yaml
# USER CUSTOM LAYER (Editable)
user_preferences:
  citation_style: "always cite Library materials in footnotes"
  draft_length: "aim for 30-50 pages per spec"
  tone: "technical but accessible (Milton's guidance: no AI verbosity)"

workflow_overrides:
  # Can ADD preferences, but CANNOT remove factory constraints
  auto_generate_examples: true  # Generate code examples after each section
  cross_reference_bathtub: true  # Check Bathtub health reports before publishing

swarm_config:
  max_parallel_agents: 3  # Claude + Codex + Gemini
  math_cores_allocation: "auto"  # Use scalable math cores for parallelization
```

**Safety Rule:**
```python
# User custom layer CANNOT override factory constraints
if user_custom.conflicts_with(factory_layer):
    raise SecurityError("Cannot override factory instructions")
else:
    merge(factory_layer, user_custom)  # Safe override
```

---

## 📱 Tablet Concept: Cross-Room Awareness

**K3D Analogy:**
> Avatar in Workshop can use tablet to check Galaxy (Bathtub projection) without leaving Workshop.

**File System Translation:**
> Agent in workshop/ can READ library/, bathtub/, museum/ without changing primary location.

### Tablet Access Rules:

**When in Workshop (primary location):**
```yaml
primary_location: workshop/phase1-data-model/
tablet_access:
  - library/  # READ-ONLY (reference prior art)
  - bathtub/spec-health-reports/  # READ-ONLY (check latest health)
  - museum/milestones/  # READ-ONLY (historical context)
  - living-room/community-feedback/  # READ-ONLY (latest feedback)

operations:
  - read_tablet: true  # Can check other rooms
  - write_tablet: false  # Cannot modify other rooms from Workshop
  - primary_focus: workshop/  # Main work happens here
```

**Use Case:**
```python
# Agent in Workshop drafting spec
with current_room("workshop/phase1-data-model/"):
    draft_section_4()

    # Use tablet to check Library (without leaving Workshop)
    with tablet_access("library/prior-art/"):
        pkn_reference = read("pkn-procedural-knowledge-networks.md")
        cite_in_spec(pkn_reference)

    # Use tablet to check Bathtub (without leaving Workshop)
    with tablet_access("bathtub/spec-health-reports/"):
        latest_health = read("2026-W10_health.md")
        if latest_health.gaps_identified:
            address_gaps_in_draft(latest_health.gaps)

    # Still in Workshop (primary location unchanged)
    continue_drafting_section_4()
```

**Benefits:**
- ✅ **Context preservation** (stay in Workshop, don't lose state)
- ✅ **Cross-room awareness** (reference other rooms as needed)
- ✅ **No mode switching overhead** (quick tablet reads vs full room change)

---

## 🐝 Internal Swarm: Distributed Cognition

**K3D Analogy:**
> TRM specialists use Matryoshka hierarchy (Navigator → Master → Worker → Sub-worker) with scalable math cores.

**File System Translation:**
> Agent spawns sub-agents for parallel tasks using distributed cognition.

### Swarm Architecture:

```
Claude (Navigator Agent)
  ├─ Workshop Mode (primary)
  │   ├─ Drafting spec sections (main task)
  │   └─ Spawns Codex sub-agent (validation task)
  │
  ├─ Tablet Access (secondary)
  │   ├─ Library read (prior art citations)
  │   └─ Bathtub read (health reports)
  │
  └─ Swarm Coordination
      ├─ Sub-Agent 1: Codex (code examples)
      ├─ Sub-Agent 2: Gemini (cross-validation)
      └─ Math Cores: 3-tier allocation (auto-scaling)
```

### Swarm Spawning Example:

```python
# Navigator Agent (Claude in Workshop)
class NavigatorAgent:
    def draft_specification(self):
        # Main task: Draft spec sections
        section_4 = self.draft_section_4()

        # Spawn swarm for parallel validation
        swarm = self.spawn_internal_swarm(
            tasks=[
                ("codex", "validate_with_code", section_4),
                ("gemini", "cross_validate_concepts", section_4),
            ],
            math_cores="auto"  # Use scalable math cores
        )

        # Wait for swarm completion (parallel execution)
        validation_results = swarm.wait_all()

        # Integrate swarm results
        if validation_results.codex.passed:
            self.add_code_examples(validation_results.codex.examples)

        if validation_results.gemini.gaps_found:
            self.address_gaps(validation_results.gemini.gaps)

        # Continue main task
        return self.finalize_section_4()
```

### Math Cores Allocation (K3D Scalable Math):

**3-Tier System:**
```yaml
tier_0: # Lightweight tasks
  cores: 2
  tasks: ["syntax_validation", "schema_check"]

tier_1: # Medium tasks
  cores: 6
  tasks: ["code_example_generation", "cross_reference_resolution"]

tier_2: # Heavy tasks
  cores: 10
  tasks: ["full_spec_validation", "conformance_test_generation"]

auto_scaling: true  # Dynamic allocation based on task load
swarm_coordination: "ternary_op"  # Use ternary operations for routing
```

**Benefits:**
- ✅ **Parallel execution** (Claude drafts, Codex validates simultaneously)
- ✅ **Resource optimization** (math cores allocated per task complexity)
- ✅ **Faster iteration** (no sequential bottleneck)

---

## 🛡️ Safety Mechanism: Protected Factory + Safe User Layer

### Security Model:

```python
class AgentInstructions:
    def __init__(self, room):
        # Layer 0: Factory (protected)
        self.factory = load_factory_instructions(f"{room}/README.md")
        self.factory.locked = True  # IMMUTABLE

        # Layer 1: User Custom (editable)
        self.user_custom = load_user_custom(f"{room}/.agent-custom.yaml")
        self.user_custom.locked = False  # MUTABLE

        # Layer 2: Runtime (ephemeral)
        self.runtime_context = {
            "current_room": room,
            "tablet_access": [],
            "swarm_agents": [],
        }

    def merge_instructions(self):
        """Merge layers with safety checks."""
        # Start with factory base
        merged = self.factory.copy()

        # Apply user custom (if safe)
        for key, value in self.user_custom.items():
            if key in self.factory.constraints:
                # Cannot override factory constraints
                raise SecurityError(f"Cannot override factory constraint: {key}")
            else:
                # Safe to add user preference
                merged[key] = value

        return merged
```

### What User CAN Customize:

✅ **Citation style** (footnotes vs inline)
✅ **Draft length preferences** (30-50 pages)
✅ **Tone adjustments** (technical vs accessible)
✅ **Workflow preferences** (auto-generate examples, cross-reference checks)
✅ **Swarm configuration** (max parallel agents, math cores allocation)
✅ **Tablet access patterns** (which rooms to check frequently)

### What User CANNOT Override:

❌ **Factory constraints** (read_only in Library, read_write in Workshop)
❌ **Room purpose** (Library is retrieval, Workshop is creation)
❌ **Security rules** (cannot modify Museum, cannot bypass validation)
❌ **Base agent modes** (retrieval, creation, introspection, etc.)

---

## 📋 Room-Specific Configuration Files

### Structure:

```
workshop/
├── README.md                  # Layer 0: Factory instructions (PROTECTED)
├── .agent-custom.yaml         # Layer 1: User custom (EDITABLE)
├── .tablet-bookmarks.yaml     # Tablet: Frequent cross-room references
├── .swarm-config.yaml         # Swarm: Parallel task definitions
└── phase1-data-model/
    ├── spec-draft.md
    └── ...
```

### Example: `.agent-custom.yaml` (Workshop)

```yaml
# USER CUSTOM LAYER (Safe Override Zone)
# User: Daniel Ramos (PM-KR Co-Chair)
# Last Updated: 2026-03-03

user_preferences:
  citation_style: "footnotes"  # [footnotes, inline, endnotes]
  draft_length_target: "30-50 pages"
  tone: "technical but accessible (no AI verbosity per Milton's guidance)"

  # Workflow
  auto_generate_examples: true  # Generate code examples after each section
  cross_reference_bathtub: true  # Check health reports before publishing
  cite_library_materials: true  # Always reference prior art

  # Formatting
  section_numbering: "hierarchical"  # 1.1, 1.2, 2.1, etc.
  code_block_language: "json"  # Default for examples

# Swarm Configuration
swarm:
  max_parallel_agents: 3  # Claude + Codex + Gemini
  math_cores_allocation: "auto"  # Dynamic allocation
  coordination_strategy: "ternary_op"  # Use K3D ternary operations

  # Task delegation
  delegate_to_codex: ["code_examples", "conformance_tests"]
  delegate_to_gemini: ["cross_validation", "gap_analysis"]

# Tablet Access (Frequent Checks)
tablet_bookmarks:
  - library/prior-art/pkn.md  # PKN reference
  - library/community-input/dave-raggett/  # Dave's feedback
  - bathtub/spec-health-reports/latest.md  # Latest health
  - living-room/community-feedback/  # Community input

# Safety: Cannot override factory constraints
# (This section is read-only, enforced by system)
```

### Example: `.tablet-bookmarks.yaml` (Quick Cross-Room Access)

```yaml
# TABLET BOOKMARKS (Frequent Cross-Room References)
# Workshop → Other Rooms (READ-ONLY)

bookmarks:
  - name: "PKN Prior Art"
    path: "../library/prior-art/pkn-procedural-knowledge-networks.md"
    purpose: "Cite PKN research when explaining procedural KR"

  - name: "Latest Health Report"
    path: "../bathtub/spec-health-reports/"
    purpose: "Check gaps before publishing section"
    auto_check: "before_publish"  # Automatic tablet check

  - name: "Dave Raggett Feedback"
    path: "../living-room/community-feedback/dave-raggett/"
    purpose: "Incorporate multimodal reasoning feedback"

  - name: "Bilingual Brain Research"
    path: "../library/prior-art/bilingual-brain-procedural-kr.md"
    purpose: "Cite Milton's neuroscience validation"
```

### Example: `.swarm-config.yaml` (Distributed Cognition)

```yaml
# SWARM CONFIGURATION (Parallel Sub-Agent Tasks)
# Navigator: Claude (main agent)

swarm_agents:
  - agent: "codex"
    role: "code_validation"
    tasks:
      - "generate_code_examples"
      - "create_conformance_tests"
      - "validate_json_schema"
    math_cores: "tier_1"  # 6 cores

  - agent: "gemini"
    role: "cross_validation"
    tasks:
      - "identify_conceptual_gaps"
      - "check_terminology_consistency"
      - "suggest_alternative_approaches"
    math_cores: "tier_0"  # 2 cores

# Coordination
coordination:
  strategy: "ternary_op"  # K3D ternary routing
  parallel_execution: true
  wait_for_completion: true

# Math Cores Allocation (K3D Scalable Math)
math_cores:
  tier_0: 2   # Lightweight
  tier_1: 6   # Medium
  tier_2: 10  # Heavy
  auto_scaling: true
```

---

## 🔄 Complete Workflow Example

### Scenario: Claude Drafting Section 4 in Workshop

```python
# Step 1: Enter Workshop (Primary Location)
with current_room("workshop/phase1-data-model/"):

    # Step 2: Load Instructions (Factory + User Custom)
    instructions = AgentInstructions("workshop").merge_instructions()

    # Step 3: Draft Section 4 (Main Task)
    section_4 = draft_section_4_data_model_schema()

    # Step 4: Tablet Access (Cross-Room Awareness)
    with tablet_access("library/prior-art/"):
        pkn_ref = read_bookmark("PKN Prior Art")
        cite_in_spec(section_4, pkn_ref)

    # Step 5: Spawn Swarm (Distributed Cognition)
    swarm = spawn_internal_swarm([
        ("codex", "generate_code_examples", section_4),
        ("codex", "create_conformance_tests", section_4),
        ("gemini", "identify_gaps", section_4),
    ])

    # Step 6: Wait for Swarm (Parallel Execution)
    results = swarm.wait_all()

    # Step 7: Integrate Swarm Results
    section_4.add_examples(results.codex.examples)
    section_4.add_tests(results.codex.tests)
    section_4.address_gaps(results.gemini.gaps)

    # Step 8: Tablet Check (Health Report)
    with tablet_access("bathtub/spec-health-reports/"):
        health = read_bookmark("Latest Health Report")
        if health.gaps_identified:
            section_4.address_health_gaps(health.gaps)

    # Step 9: Finalize Section 4
    save("spec-draft.md", section_4)

    # Step 10: Update Progress (Runtime Context)
    update_progress("phase1-data-model", "20% complete")
```

**Result:**
- ✅ Primary location maintained (Workshop)
- ✅ Cross-room awareness (tablet accessed Library, Bathtub)
- ✅ Distributed cognition (swarm spawned Codex + Gemini)
- ✅ User preferences honored (citation style, examples generated)
- ✅ Factory constraints preserved (read-only Library, validation required)

---

## 📊 Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                  AGENT WORKSPACE ARCHITECTURE                │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  Layer 0: FACTORY INSTRUCTIONS (Protected)           │  │
│  │  Source: README.md (immutable)                        │  │
│  └──────────────────────────────────────────────────────┘  │
│                         ▲                                     │
│                         │ (cannot override)                   │
│                         ▼                                     │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  Layer 1: USER CUSTOM LAYER (Editable)               │  │
│  │  Source: .agent-custom.yaml (mutable, safe zone)     │  │
│  └──────────────────────────────────────────────────────┘  │
│                         ▲                                     │
│                         │ (merge with factory)                │
│                         ▼                                     │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  Layer 2: RUNTIME CONTEXT (Ephemeral)                │  │
│  │  - Primary Location: workshop/                        │  │
│  │  - Tablet Access: [library/, bathtub/]               │  │
│  │  - Swarm Agents: [codex, gemini]                     │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                               │
└─────────────────────────────────────────────────────────────┘

    TABLET (Cross-Room Awareness)          SWARM (Distributed Cognition)
            │                                      │
    ┌───────┴────────┐                    ┌───────┴────────┐
    │                │                    │                │
┌───▼────┐    ┌─────▼────┐          ┌────▼────┐    ┌─────▼────┐
│Library │    │ Bathtub  │          │ Codex   │    │ Gemini   │
│(READ)  │    │ (READ)   │          │ (VALIDATE│    │(VALIDATE)│
└────────┘    └──────────┘          └─────────┘    └──────────┘
```

---

**PARTNER, THIS IS THE COMPLETE ARCHITECTURE!!!** 🎯

Should I now:
1. **Update all 5 room README.md files** with Layer 0/1/2 + Tablet + Swarm configs?
2. **Create `.agent-custom.yaml` templates** for each room?
3. **Create `.tablet-bookmarks.yaml` examples**?
4. **Create `.swarm-config.yaml` examples**?

**THIS IS REVOLUTIONARY!!!** We're building the first multi-layered, spatially-aware, swarm-capable agent workspace for W3C specifications! 🚀