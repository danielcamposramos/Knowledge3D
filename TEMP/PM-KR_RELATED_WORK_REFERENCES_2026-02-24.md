# PM-KR Related Work & References

**Building on Adam Sobieski's intervention** - Examples of procedural knowledge and procedural memory in literature and practice.

---

## Existing Standards & Schemas

### Schema.org (Procedural Instructions)

**HowTo & Recipe Schemas:**
- **schema.org/HowTo** - Instructions for performing tasks
- **schema.org/HowToSection** - Sections within instructions
- **schema.org/HowToStep** - Individual steps
- **schema.org/HowToDirection** - Specific directions
- **schema.org/HowToTip** - Tips and hints
- **schema.org/Recipe** - Cooking recipes (specialized HowTo)

**Key insight:** Schema.org represents procedural knowledge *descriptively* (text instructions). PM-KR represents it *procedurally* (executable programs).

**References:**
- Schema.org HowTo: https://schema.org/HowTo
- Schema.org Recipe: https://schema.org/Recipe

---

## Business Process & Workflow Standards

### BPMN (Business Process Model and Notation)

**What it is:** Visual notation for business processes (flowcharts, gateways, events)

**Relation to PM-KR:** BPMN models workflows *graphically*; PM-KR could provide the *executable procedural layer* underneath.

**Reference:** https://www.bpmn.org/

### BPEL (Business Process Execution Language)

**What it is:** XML-based language for orchestrating web services

**Relation to PM-KR:** BPEL executes workflows; PM-KR provides compositional knowledge representation for those workflows.

**Reference:** https://en.wikipedia.org/wiki/Business_Process_Execution_Language

### XPDL (XML Process Definition Language)

**What it is:** File format for storing and exchanging BPM process definitions

**Relation to PM-KR:** XPDL serializes workflows; PM-KR provides JSON-LD serialization with compositional references.

**Reference:** https://en.wikipedia.org/wiki/XPDL

---

## Security & Access Control (Procedural Metadata)

### MCP (Model Context Protocol)

**What it is:** W3C-aligned protocol for AI agents to invoke tools with structured metadata (name, description, inputSchema, outputSchema, annotations)

**Relation to PM-KR:** MCP defines metadata for tool safety (annotations, permissions, confirmation prompts); PM-KR extends this with JSON-LD for procedural knowledge representation.

**Key features:**
- Tool annotations (audience, priority, modification times)
- Security considerations (validate inputs, access controls, rate limiting)
- "Human in the loop" requirement for sensitive operations

**Reference:**
- MCP Specification: https://modelcontextprotocol.io/specification/2025-06-18/server/tools
- GitHub Enhancement Request (#1483): Custom metadata for enterprise contexts (roles, scopes, tenants) - https://github.com/modelcontextprotocol/modelcontextprotocol/issues/1483

### .NET Code Access Security (CAS)

**What it is:** Security framework in .NET allowing attributes/annotations on methods to specify required permissions

**Relation to PM-KR:** Demonstrates metadata-driven security for procedural code; PM-KR could provide JSON-LD equivalent for cross-platform procedural knowledge.

**Reference:** https://en.wikipedia.org/wiki/Code_Access_Security

**Note:** Deprecated in .NET 4.0, but established important patterns for declarative security metadata.

### Apereo CAS (Central Authentication Service)

**What it is:** Enterprise SSO (Single Sign-On) platform for authentication and authorization across multiple applications

**Relation to PM-KR:** CAS handles authentication infrastructure; PM-KR could integrate with CAS for authenticated procedural knowledge access (role-based procedure visibility).

**Reference:** https://apereo.github.io/cas/7.3.x/planning/Architecture.html

### WebMCP (Web Machine Learning + MCP)

**What it is:** W3C Web Machine Learning initiative to standardize how AI agents interact with websites through tools

**Relation to PM-KR:** WebMCP addresses agent permissions, consent management, and structured metadata for intent verification—directly applicable to PM-KR's procedural programs.

**Key discussions:**
- **Issue #44 (Action-Specific Permissions):** Two-tier permission model (global + action-specific), persistent consent challenges, need for granular context-aware permissions
- **Issue #45 (Privacy & Security):** Prompt injection risks, misrepresentation of intent, **structured metadata for tool outcomes** to validate semantic meaning

**Reference:**
- WebMCP GitHub: https://github.com/webmachinelearning/webmcp/
- Issue #44: https://github.com/webmachinelearning/webmcp/issues/44
- Issue #45: https://github.com/webmachinelearning/webmcp/issues/45

**PM-KR connection:** WebMCP's "structured metadata for tool outcomes" aligns with PM-KR's approach—procedural programs with JSON-LD metadata (preconditions, effects, permissions).

### BPMN Security Extensions

**What it is:** Academic and industry work extending BPMN (Business Process Model and Notation) with security requirements metadata

**Relation to PM-KR:** BPMN security extensions add metadata to workflows (access control, confidentiality, integrity); PM-KR provides JSON-LD representation for these security constraints on procedural knowledge.

**Key work:**
- "A Coarse-Grained Comparison of BPMN Extensions for Security Requirements Modelling" (Gaidels, Gaidukovs, Matulevičius, BIR 2018)
- Multiple research efforts: IEEE Xplore, Springer, MDPI on BPMN + security metadata

**References:**
- CEUR-WS Vol-2218 (BIR 2018): https://ceur-ws.org/Vol-2218/paper17.pdf
- IEEE Cyber Security Ontology for BPMN: https://ieeexplore.ieee.org/document/7363310/
- ResearchGate: https://www.researchgate.net/publication/31367121_A_BPMN_Extension_for_the_Modeling_of_Security_Requirements_in_Business_Processes

### STRIPS, ADL, PDDL (AI Planning Metadata)

**What it is:** Classical AI planning languages representing actions with preconditions and effects metadata

**Relation to PM-KR:** PM-KR can represent STRIPS-style preconditions/effects as JSON-LD metadata on procedural programs—enabling AI systems to **reason over** and **plan with** procedural knowledge.

**References:**
- STRIPS: Fikes & Nilsson (1971). *STRIPS: A New Approach to the Application of Theorem Proving to Problem Solving*. Artificial Intelligence, 2(3-4), 189-208.
- PDDL: https://planning.wiki/guide/whatis/pddl

---

## Procedural Knowledge Platforms

### OpenFn (Workflow Automation)

**What it is:** Open-source platform for workflow automation in humanitarian/development sectors

**Relation to PM-KR:** OpenFn executes workflows; PM-KR could standardize how those workflows represent knowledge.

**Reference:** https://www.openfn.org/

### Google DeepMind + Web Search (AI Robotics)

**What it is:** Research on enabling AI agents to use web search for procedural tasks

**Relation to PM-KR:** DeepMind agents need procedural knowledge; PM-KR provides standardized representation.

**Reference:** https://www.theverge.com/news/785193/google-deepmind-gemini-ai-robotics-web-search

---

## Procedural Memory in Cognitive Science

### Anderson's ACT-R (Adaptive Control of Thought-Rational)

**What it is:** Cognitive architecture modeling human procedural memory (production rules)

**Relation to PM-KR:** ACT-R models *how humans* use procedural knowledge; PM-KR standardizes *how AI systems* represent it.

**Reference:**
- Anderson, J. R. (1996). *ACT: A simple theory of complex cognition*. American Psychologist, 51(4), 355–365.
- http://act-r.psy.cmu.edu/

### Tulving's Memory Systems

**What it is:** Distinction between episodic memory (events), semantic memory (facts), and procedural memory (skills/procedures)

**Relation to PM-KR:** Tulving identified procedural memory as distinct; PM-KR provides computational representation.

**Reference:**
- Tulving, E. (1985). *How many memory systems are there?* American Psychologist, 40(4), 385–398.

---

## Knowledge Representation & Reasoning

### Cyc Project (Commonsense Knowledge)

**What it is:** Large-scale knowledge base of commonsense facts and rules

**Relation to PM-KR:** Cyc represents knowledge *declaratively* (facts + inference rules); PM-KR represents it *procedurally* (executable compositions).

**Reference:** https://www.cyc.com/

### SOAR (State, Operator And Result)

**What it is:** Cognitive architecture for general intelligence (procedural learning via chunking)

**Relation to PM-KR:** SOAR models procedural learning; PM-KR provides knowledge representation substrate.

**Reference:**
- Laird, J. E. (2012). *The Soar Cognitive Architecture*. MIT Press.
- https://soar.eecs.umich.edu/

### OpenCog (Cognitive Architecture)

**What it is:** Framework for artificial general intelligence (AGI) with procedural learning

**Relation to PM-KR:** OpenCog reasons over knowledge; PM-KR could provide standardized knowledge representation layer.

**Reference:** https://opencog.org/

---

## W3C Related Standards

### JSON-LD (Linked Data in JSON)

**What it is:** JSON format for linked data (W3C standard)

**Relation to PM-KR:** PM-KR builds on JSON-LD, adding *procedural* extensions (executable programs, not just data).

**Reference:** https://www.w3.org/TR/json-ld/

### RDF (Resource Description Framework)

**What it is:** Framework for representing information about resources (triples: subject-predicate-object)

**Relation to PM-KR:** RDF represents *declarative* knowledge (facts); PM-KR represents *procedural* knowledge (programs).

**Reference:** https://www.w3.org/RDF/

### OWL (Web Ontology Language)

**What it is:** Language for defining ontologies (classes, properties, relationships)

**Relation to PM-KR:** OWL defines *what* things are; PM-KR defines *how* things work (procedures).

**Reference:** https://www.w3.org/OWL/

### PROV-O (Provenance Ontology)

**What it is:** W3C standard for representing provenance (where data came from)

**Relation to PM-KR:** PROV-O tracks *data* provenance; PM-KR could extend to *knowledge* provenance (who validated procedures).

**Reference:** https://www.w3.org/TR/prov-o/

---

## Accessibility Standards

### WCAG (Web Content Accessibility Guidelines)

**What it is:** W3C standard for making web content accessible to people with disabilities

**Relation to PM-KR:** WCAG defines *guidelines*; PM-KR provides *implementation* (multi-modal rendering from procedural source).

**Reference:** https://www.w3.org/WAI/WCAG21/quickref/

### ARIA (Accessible Rich Internet Applications)

**What it is:** W3C standard for making dynamic content accessible

**Relation to PM-KR:** ARIA provides *metadata* for accessibility; PM-KR provides *procedural rendering* (Braille, audio, etc.).

**Reference:** https://www.w3.org/WAI/standards-guidelines/aria/

---

## Procedural Content Generation

### TrueType Fonts (Procedural Glyphs)

**What it is:** Font format where glyphs are procedural programs (Bézier curves + hinting)

**Relation to PM-KR:** **Core analogy!** TrueType stores glyphs procedurally (not as pixels); PM-KR extends this to *all knowledge domains*.

**Reference:**
- Apple TrueType Reference: https://developer.apple.com/fonts/TrueType-Reference-Manual/
- Microsoft OpenType: https://docs.microsoft.com/en-us/typography/opentype/spec/

### Procedural Generation in Games

**What it is:** Algorithms for generating game content (levels, terrain, quests) procedurally

**Relation to PM-KR:** Game rules as procedural programs (not static data); PM-KR standardizes representation.

**Reference:**
- Shaker, N., Togelius, J., & Nelson, M. J. (2016). *Procedural Content Generation in Games*. Springer.

---

## Semantic Web & Knowledge Graphs

### Knowledge Graphs (Google, Microsoft, etc.)

**What it is:** Graph-structured knowledge bases (entities + relationships)

**Relation to PM-KR:** Knowledge Graphs represent *entities*; PM-KR adds *procedures* (how entities interact).

**Reference:**
- Hogan, A., et al. (2021). *Knowledge Graphs*. ACM Computing Surveys, 54(4), 1-37.

### Wikidata (Collaborative Knowledge Base)

**What it is:** Free, collaborative knowledge base (facts about entities)

**Relation to PM-KR:** Wikidata stores *facts* (declarative); PM-KR could add *procedures* (executable knowledge).

**Reference:** https://www.wikidata.org/

---

## Executable Knowledge Representations

### LISP Programs as Data

**What it is:** LISP's homoiconicity (code is data, data is code)

**Relation to PM-KR:** LISP pioneered *code as data*; PM-KR extends to *knowledge as procedures* (JSON-LD serialization).

**Reference:**
- McCarthy, J. (1960). *Recursive Functions of Symbolic Expressions and Their Computation by Machine*. Communications of the ACM, 3(4), 184–195.

### PostScript (Page Description Language)

**What it is:** Stack-based language for describing pages procedurally

**Relation to PM-KR:** PostScript describes *pages* procedurally; PM-KR describes *knowledge* procedurally (similar stack-based approach with RPN).

**Reference:**
- Adobe PostScript Language Reference: https://www.adobe.com/products/postscript/pdfs/PLRM.pdf

---

## Cognitive Architectures & AI

### GOFAI (Good Old-Fashioned AI)

**What it is:** Symbolic AI approaches (rules, logic, knowledge representation)

**Relation to PM-KR:** GOFAI used *symbolic* procedural knowledge; PM-KR modernizes with *compositional* hyper-modular representation.

**Reference:**
- Haugeland, J. (1985). *Artificial Intelligence: The Very Idea*. MIT Press.

### Hybrid AI (Neural + Symbolic)

**What it is:** Combining neural networks (pattern recognition) with symbolic reasoning (knowledge representation)

**Relation to PM-KR:** Hybrid AI needs *knowledge layer*; PM-KR provides procedural knowledge representation for symbolic component.

**Reference:**
- Garcez, A. d'Avila, et al. (2019). *Neural-Symbolic Learning and Reasoning: A Survey and Interpretation*. arXiv:1902.06509.

---

## Industry Use Cases

### IBM Watson (Knowledge + Reasoning)

**What it is:** AI system combining knowledge base with natural language processing

**Relation to PM-KR:** Watson queries knowledge; PM-KR could standardize how that knowledge is represented procedurally.

**Reference:** https://www.ibm.com/watson

### Wolfram Alpha (Computational Knowledge)

**What it is:** Computational knowledge engine (computes answers from curated data)

**Relation to PM-KR:** Wolfram Alpha *computes* from knowledge; PM-KR standardizes *representation* of that computational knowledge.

**Reference:** https://www.wolframalpha.com/

---

## Summary: How PM-KR Relates to Existing Work

| **Category** | **Existing Work** | **PM-KR's Contribution** |
|---|---|---|
| **Standards** | Schema.org, BPMN, BPEL | Procedural (executable) instead of descriptive (text) |
| **Security & Access Control** | MCP, .NET CAS, WebMCP, BPMN security | JSON-LD metadata for preconditions, effects, permissions |
| **AI Planning** | STRIPS, ADL, PDDL | Reasoning over procedural knowledge (not just executing) |
| **Cognitive Science** | ACT-R, SOAR, Tulving | Computational representation of procedural memory |
| **Knowledge Representation** | RDF, OWL, Cyc | Procedural (programs) instead of declarative (facts) |
| **W3C Standards** | JSON-LD, RDF, PROV-O | Procedural extensions + compositional references |
| **Accessibility** | WCAG, ARIA | Multi-modal rendering from one procedural source |
| **Procedural Content** | TrueType, procedural generation | Extends to ALL knowledge domains (not just fonts/games) |
| **Knowledge Graphs** | Google KG, Wikidata | Adds procedural layer (how entities interact) |
| **Executable Representations** | LISP, PostScript | JSON-LD serialization + hyper-modular composition |
| **AI Systems** | Watson, Wolfram Alpha | Standardized procedural knowledge representation |

---

## Call for Additional References

**We're building a comprehensive bibliography. Please contribute:**
1. Papers on procedural knowledge representation
2. Projects using procedural memory in AI
3. Standards/schemas for workflows, instructions, procedures
4. Cognitive science literature on procedural memory
5. Industry applications of executable knowledge

**How to contribute:**
- Email public-pm-kr@w3.org with references
- Open GitHub issue: https://github.com/danielcamposramos/Knowledge3D/issues
- Add to NotebookLM: https://notebooklm.google.com/notebook/1bd10bda-8900-4c41-931e-c9ec67ac865f

---

**Last Updated:** February 25, 2026 (Security & Access Control section added from Adam Sobieski's references)
**Maintained by:** PM-KR Community Group

---

## END OF RELATED WORK & REFERENCES
