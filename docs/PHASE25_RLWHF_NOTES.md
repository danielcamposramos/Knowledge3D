# Phase 25 RLWHF Training Notes

## Teacher Prompt Update
- The RLWHF teacher now addresses the controller as **K3D Automatic RLWHF Training Program**.
- All references to “AI response” were renamed to “student response” so evaluation feedback focuses on the student being trained.
- The teacher always receives the original question and the expected answer alongside the student’s reply.

## Corpus Generation Hygiene
- All Phase 25 corpus builders skip sentences containing authorship, publishing, or copyright boilerplate (e.g., “ISBN”, “All rights reserved”, publisher names).
- This keeps drill material focused on conceptual content, avoiding trivia about publication metadata.

## Training Loop Adjustments
- `AlgorithmicThinkingTrainer` now supplies question + expected answer to both exaone models for each RLWHF check.
- Teacher evaluations therefore operate with full context, eliminating guesses when scoring.

## Next Steps
1. Rebuild the corpora with the updated filters (already done as part of this change).
2. Relaunch the PTX + RLWHF training loop and monitor `logs/phase25_thinking_train.log` once exaone is ready.
