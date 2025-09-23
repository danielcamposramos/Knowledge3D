# Training & Consolidation Directives (Sep 2025)

This note captures the latest guidance for steering Phase 25/meaning-cluster training toward a fully embodied AGI MVP.

## Prompt Hygiene

- **Cull nonsense prompts**: questions like “What significant event happened in 000?” or similar auto-generated fragments must be removed from future drill sets. Curate prompts to emphasise meaning-oriented reasoning mixed with traditional recall.
- **Retire mastered prompts**: once the fused head + tablet confirmation answer a prompt correctly in two sessions, remove it from the next run. Archive it for potential reuse or expansion, but keep the active queue focused on unsolved or growing concepts.

## Sleep-Time Policy

- **Always absorb trusted teacher feedback**: during learning mode, log every teacher correction—full points or half points—and let SleepTime consolidate immediately. Knowledge should flow House-first; museum relocation handles deprecated artifacts.
- **Timestamp everything**: tag training runs, prompt logs, and consolidated artifacts with timezone-aware timestamps so we can trace learning cadence.

## Mode Separation

- **Learning vs Inference**: treat lesson mode (house/classroom) separately from inference mode. Self-reflections still run during inference but must carry a critical tone; tests should not toggle lesson mode midstream.
- **Inference compute time**: expose a thinking tag / inference compute loop even outside training. This keeps the reasoning trace visible and lets us begin tagging thought patterns ASAP.

## Memory Flow & Tablet Use

- The avatar lives inside the House. The tablet provides a galaxy-standard view only when extra context is needed. Use it to:
  - Map house artifacts into temporary thinking memory.
  - Launch tool sessions (MCP, browser/VMs) and capture transcripts.
  - Trigger on-demand loads for extended reasoning, then consolidate via SleepTime.

## Thematic Expansion

- **Time mastery**: broaden training data on time—machine cycles, human calendars, temporal reasoning—using the curated JSON corpus under `/mnt/arquivos/0 ChatGPTs/DataBase/EchoSystems Default Libraries/Understand Time/JSON/` and expand it with exaone3.5 / exaone-deep when crafting prompts.
- **Math enrichment**: add drills from `/mnt/arquivos/0 ChatGPTs/DataBase/EchoSystems Default Libraries/Advanced Maths/BasicMath/JSON/` and `/mnt/arquivos/0 ChatGPTs/DataBase/EchoSystems Default Libraries/Advanced Maths/Financial Math/JSON/`, again leveraging exaone models for augmentation.

## Next Actions Summary

1. Clean the prompt pool (remove nonsensical items).
2. Update trainers to mark prompts as “mastered” after two perfect runs and exclude them from the next session.
3. Ensure every training run writes a timezone-aware timestamp.
4. Let SleepTime absorb teacher feedback every cycle, without waiting for manual triggers.
5. Surface inference compute/thinking tags during non-training inference.
6. Ingest additional time and math corpora and feed them through the House-first paradigm.
