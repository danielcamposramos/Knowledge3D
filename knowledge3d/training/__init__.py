"""Training subpackage for Knowledge3D.

This package contains active training-time curricula (ARC-AGI, math benchmarks,
RLWHF). Archived curriculum-specific training code lives under
Old_Attempts/curriculum_specific_training. Runtime inference should remain
sovereign (PTX + RPN + Galaxy); training/ingestion tooling may use richer
Python-side pipelines.
"""
