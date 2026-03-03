# Comparison Chart: PM-KR vs Common AI Knowledge Patterns

| Dimension | Traditional LLMs | RAG Pipelines | PM-KR Procedural Memory |
|-----------|------------------|---------------|-------------------------|
| Core storage model | Knowledge largely internal to model weights | External retrieval + model synthesis | Canonical procedural nodes + references |
| Typical model size reference | 100B to 1T+ parameter classes | Depends on downstream model | About 7M parameter runtime core in K3D reports |
| Latency transparency | Opaque internals, hard to inspect stepwise | Partially visible prompts and retrieved chunks | Deterministic operation traces for procedural execution |
| Explainability mode | Mostly post-hoc narrative | Retrieval context visible, synthesis often opaque | Hard traceability through executable steps and references |
| Composability | Monolithic weight updates | Prompt chains and retriever orchestration | Layered form->meaning->rules->meta-rules composition |
| Duplication pressure | High data replication across training and serving pipelines | Moderate to high across indexes and snapshots | Reference-first model designed to reduce duplicate payload |
| Human and AI source parity | Usually indirect | Partial | Explicit dual-client contract orientation |
| Governance controls | Policy wrappers around model behavior | Policy plus retrieval guardrails | Boundary contracts plus auditable procedural lineage |
| Carbon profile tendency | High for large-scale training/inference regimes | Moderate to high depending on model usage | Designed for lower compute duplication; uses scenario modeling |
| Standards posture | Vendor-defined interfaces dominate | Framework-specific implementations | W3C Community Group standardization path in progress |

Notes:
- The PM-KR values are based on K3D published internal reports and W3C package drafts.
- Carbon figures in this press kit are scenario projections and should be presented as modeled outcomes, not guaranteed baselines.

Sources:
- README.md
- docs/W3C/PM_KR_NORMATIVE_MODEL.md
- docs/W3C/PM_KR_EVIDENCE_VALIDATION_MATRIX.md
- docs/CARBON_BLUEPRINT_10_YEAR_PROJECTION.md
