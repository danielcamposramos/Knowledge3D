# Teacher-Student Distillation and Contrastive Learning for TRM

Date: 2026-04-06  
Prepared for: Claude and Daniel  
Prepared by: Codex

## Executive Summary

This report connects three threads:

1. the local `AI-RLWHF` teacher-student and contrastive-honesty work,
2. current public research on teacher-student distillation and contrastive learning for modern language models,
3. the Tiny Recursive Model (TRM) direction now relevant to Knowledge3D.

The main conclusion is direct:

- **teacher-student training** can improve TRM by transferring capability from stronger frontier or mid-sized teachers without forcing TRM itself to become large;
- **contrastive learning** can improve TRM by turning each teacher-student interaction into multiple supervised signals instead of one scalar reward;
- for TRM specifically, the highest-value extension is not generic next-token distillation, but **recursive-trace distillation + uncertainty-aware contrastive training + memory-state contrastive shaping**.

Put simply:

> TRM should not only learn the final answer. It should learn what a good recursive trajectory looks like, what an honest uncertainty signal looks like, and which internal states should be closer or farther apart in embedding space.

That combination is highly compatible with both `AI-RLWHF` and K3D's sovereign long-term direction.

## Scope and Sources

### Local sources used

Primary local source:
- [README.md](/mnt/arquivos/EchoSystems%20AI%20Studios/Knowledge%203D%20Standard/GitHub/AI-RLWHF/README.md)
- [rlwhf-framework.md](/mnt/arquivos/EchoSystems%20AI%20Studios/Knowledge%203D%20Standard/GitHub/AI-RLWHF/docs/rlwhf-framework.md)
- [CONTRASTIVE_HONESTY_LEARNING_SPECIFICATION.md](/mnt/arquivos/EchoSystems%20AI%20Studios/Knowledge%203D%20Standard/GitHub/AI-RLWHF/docs/CONTRASTIVE_HONESTY_LEARNING_SPECIFICATION.md)
- [CONTRASTIVE_HONESTY_IMPLEMENTATION_REPORT.md](/mnt/arquivos/EchoSystems%20AI%20Studios/Knowledge%203D%20Standard/GitHub/AI-RLWHF/docs/CONTRASTIVE_HONESTY_IMPLEMENTATION_REPORT.md)
- [contrastive_loss.py](/mnt/arquivos/EchoSystems%20AI%20Studios/Knowledge%203D%20Standard/GitHub/AI-RLWHF/plugins/core/contrastive_loss.py)
- [LLM_ENHANCEMENT_SUMMARY.md](/mnt/arquivos/EchoSystems%20AI%20Studios/Knowledge%203D%20Standard/GitHub/AI-RLWHF/LLM_ENHANCEMENT_SUMMARY.md)

Relevant K3D-local context:
- [CODEX_HANDOFF_E64_TO_PRESENT_2026-04-06.md](/mnt/arquivos/EchoSystems%20AI%20Studios/Knowledge%203D%20Standard/GitHub/Knowledge3D/docs/Briefings/CODEX_HANDOFF_E64_TO_PRESENT_2026-04-06.md)
- [KNOWLEDGE_PROCEDURALIZER_SPECIFICATION.md](/mnt/arquivos/EchoSystems%20AI%20Studios/Knowledge%203D%20Standard/GitHub/Knowledge3D/docs/vocabulary/KNOWLEDGE_PROCEDURALIZER_SPECIFICATION.md)
- [KNOWLEDGEVERSE_SPECIFICATION.md](/mnt/arquivos/EchoSystems%20AI%20Studios/Knowledge%203D%20Standard/GitHub/Knowledge3D/docs/vocabulary/KNOWLEDGEVERSE_SPECIFICATION.md)

### External primary/public sources used

- Hinton, Vinyals, Dean, "Distilling the Knowledge in a Neural Network" (2015): https://arxiv.org/abs/1503.02531
- Wang et al., "MiniLM: Deep Self-Attention Distillation for Task-Agnostic Compression of Pre-Trained Transformers" (2020): https://arxiv.org/abs/2002.10957
- Gao, Yao, Chen, "SimCSE: Simple Contrastive Learning of Sentence Embeddings" (2021): https://arxiv.org/abs/2104.08821
- Tan et al., "GKD: A General Knowledge Distillation Framework for Large-scale Pre-trained Language Model" (2023): https://arxiv.org/abs/2306.06629
- Fisch et al., "Robust Preference Optimization through Reward Model Distillation" (2024): https://arxiv.org/abs/2405.19316
- Song and Zheng, "A Survey of On-Policy Distillation for Large Language Models" (2026): https://arxiv.org/abs/2604.00626
- Jiang et al., "Mixtral of Experts" (2024): https://arxiv.org/abs/2401.04088
- DeepSeek-AI, "DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model" (2024): https://arxiv.org/abs/2405.04434
- He et al., "HMT: Hierarchical Memory Transformer for Efficient Long Context Language Processing" (2024): https://arxiv.org/abs/2405.06067
- Jolicoeur-Martineau, "Less is More: Recursive Reasoning with Tiny Networks" (2025): https://arxiv.org/abs/2510.04871

### Missing local source

I could not locate a sibling repository matching `Scalar_Wizzard`, `Scalar_Wizard`, or obvious spelling variants under the current workspace roots.  
This report therefore uses `AI-RLWHF` plus public literature and current K3D/TRM context. If the second repo lives elsewhere, it should be added in a later revision.

## What AI-RLWHF Already Contributes

The most important thing in `AI-RLWHF` is that it already moves beyond simple scalar RLHF.

### 1. Teacher-student architecture is explicit

From the local RLWHF framework:
- the **teacher** is a stronger evaluator model,
- the **student** is the model under training,
- every interaction stores prompt, student answer, teacher critique, and reward,
- the loop is designed for replayable, auditable training.

This matters because it already matches the central distillation logic in modern LLM training:
- student trajectories are generated,
- a stronger teacher judges or guides them,
- feedback is persisted in structured form,
- the student updates against that structured signal.

That is already much closer to modern **on-policy distillation** than to classic offline supervised fine-tuning.

### 2. Contrastive honesty learning is stronger than scalar reward-only training

The key local innovation in `AI-RLWHF` is the decomposition of each response into:
- correctness signals,
- honesty signals,
- cross-response contrast signals.

This is better than a single scalar for a simple reason:
- a `+1` answer contains correct fragments, wrong fragments, and an honesty behavior,
- a `-1` answer also contains mixed correctness plus missing honesty,
- scalar-only optimization throws away that structure,
- contrastive decomposition preserves it.

The local specification is right to argue that this multiplies useful signal per example.

For TRM, this is especially important because TRM is small. Small models waste training budget faster when supervision is collapsed into overly coarse labels.

## Why This Maps Cleanly Onto Current LLM Architectures

The phrase "current LLM architectures" should not be read narrowly as only one model family. Teacher-student and contrastive learning apply differently across several active architecture classes.

### 1. Dense decoder-only Transformers

Teacher-student learning already fits the standard decoder-only stack well:
- logits can be distilled,
- hidden states can be distilled,
- attention patterns can be distilled,
- sequence-level preferences can be distilled.

The MiniLM line is especially relevant because it showed that distilling **self-attention structure** can preserve much more capability than naive output matching alone.

Implication for TRM:
- if TRM is used as a compact recursive core, then **state-transition structure** is the analogue of attention structure in a Transformer;
- we should distill not just output tokens, but recursive update geometry.

### 2. Sparse MoE LLMs

Modern MoE systems such as Mixtral and DeepSeek-V2 make selective computation central:
- only part of the total parameter mass is active per token,
- routing quality matters,
- efficient specialization matters as much as raw parameter count.

This is highly relevant to K3D because the K3D/TRM direction already assumes:
- a small core,
- specialized pathways,
- explicit routing,
- externally structured memory.

Implication for TRM:
- teacher-student learning should supervise **route choice**, not just answer choice;
- contrastive losses can separate "good route / bad route" internal trajectories.

### 3. Long-context / memory-augmented LLMs

Architectures like HMT and other memory-augmented systems address the fact that raw context windows are not enough.

K3D is already beyond flat context windows:
- Galaxy and House are persistent structured memory,
- the model is supposed to query and update that memory,
- reasoning is supposed to be inspectable.

Implication for TRM:
- the right distillation target is not merely language output,
- it is **memory access behavior**:
  - what should be retrieved,
  - what should be ignored,
  - what should be consolidated,
  - what uncertainty should be surfaced when memory is missing.

### 4. On-policy distillation and black-box teacher regimes

The 2026 OPD survey is especially relevant because it frames the key weakness of static teacher datasets:
- off-policy distillation trains on frozen teacher outputs,
- inference then fails on student-induced errors,
- on-policy distillation lets the student generate its own trajectories and receive feedback on those trajectories.

This is almost exactly the right regime for TRM:
- TRM should generate its own recursive passes,
- the teacher should assess those passes,
- the student should learn on the mistakes it actually makes.

That is a better fit than static imitation.

## Why Contrastive Learning Matters Even More For TRM Than For Large LLMs

TRM is small. That changes the economics of supervision.

Large models can sometimes brute-force through noisy scalar supervision because they have excess representational slack. A tiny recursive model cannot.

Contrastive learning helps TRM in four specific ways:

### 1. Better sample efficiency

A single answer can generate:
- positive fragments,
- negative fragments,
- better/worse response pairs,
- honest/dishonest uncertainty signals.

For a small model, this is close to free extra supervision.

### 2. Better internal geometry

Contrastive objectives do not just reward outputs. They shape representation space.

For TRM, that means:
- honest uncertainty states can be made closer to correct cautious reasoning than to confident fabrication,
- recursive states that are semantically compatible can cluster,
- route states that lead to hallucination can be pushed away.

This is likely more important than next-token imitation for a small recursive model.

### 3. Better calibration

The local RLWHF insight that `"I don't know"` should outrank confident fabrication is strategically correct.

Contrastive honesty learning gives TRM a way to learn:
- that uncertainty is not failure,
- that overconfident wrongness is structurally worse than admitted ignorance,
- that calibration itself is a learnable axis.

This matters for benchmark safety and for real-system trust.

### 4. Better decomposition of reasoning

TRM is recursive. Recursion creates intermediate states.

Contrastive learning can supervise:
- which intermediate states look like improvement,
- which look like drift,
- which look like unresolved ambiguity,
- which look like fabricated closure.

This is exactly where a tiny recursive model can gain disproportionally.

## The Most Important Conceptual Shift

The right teacher-student framing for TRM is not:

> teacher gives answer, student copies answer

It is:

> teacher evaluates the student's recursive trajectory, decomposition, uncertainty behavior, and final answer together

That is a much better fit with both:
- `AI-RLWHF`
- the TRM paper's core claim that recursive structure matters

## Concrete Ways To Improve TRM

The following are the most defensible technical proposals.

### Proposal 1. Recursive-Trace Distillation

Instead of distilling only final outputs, distill the sequence:
- initial hypothesis
- intermediate recursive updates
- final answer
- uncertainty state

Teacher annotations should label:
- which recursive step improved the answer,
- which step introduced an error,
- whether the final state should have remained unresolved.

Why this matters:
- it converts TRM's recursion from an internal mystery into a supervised object,
- it is the recursive analogue of hidden-state or attention distillation.

### Proposal 2. Contrastive Honesty Loss For TRM State Space

Use a projector head over TRM internal states and apply a three-axis contrastive loss:
- correctness axis
- honesty axis
- better/worse recursive trajectory axis

Positive pairs:
- recursive states that move toward correct reasoning,
- honest uncertainty states,
- trajectories that correctly stop when unresolved.

Negative pairs:
- fabrication states,
- overconfident wrong closures,
- route states that bypass necessary evidence.

Why this matters:
- it teaches TRM not just what to output, but what internal configuration is desirable.

### Proposal 3. Teacher-Assistant Cascades

Classic distillation often benefits from teacher-assistant staging, especially when the teacher-student gap is very large.

For TRM this is probably critical:
- frontier teacher or strong API teacher
- mid-size student / adapter teacher
- tiny recursive student

Why this matters:
- directly distilling a very large model into a tiny recursive model can be too abrupt,
- staged distillation can preserve structure better.

### Proposal 4. Route-Level Distillation

K3D already depends on routing across families such as `QUESTION`, `MATH`, `GENERAL`, and `GRAMMAR`.

Teacher-student supervision should explicitly train:
- route selection,
- route depth,
- evidence sufficiency,
- materialization readiness.

This is the equivalent of distilling control flow.

Why this matters:
- many observed failures in K3D were not only "wrong answer" failures,
- they were wrong-route, shallow-route, or validator-without-materialization failures.

### Proposal 5. Memory-Aware Contrastive Retrieval

Because K3D has explicit persistent memory, teacher feedback can supervise retrieval itself.

Positive retrieval examples:
- stars, rules, or facts that should have been accessed

Negative retrieval examples:
- irrelevant but superficially similar stars
- anti-pattern routes
- source-title leakage

Why this matters:
- it couples contrastive learning to the actual persistent substrate,
- which is more aligned with K3D than generic Transformer fine-tuning.

### Proposal 6. Reward Model Distillation For Uncertainty

The reward-model distillation literature is relevant because it preserves richer preference distributions than pairwise labels alone.

For TRM, the distilled reward should explicitly include:
- factual correctness
- honesty / uncertainty calibration
- procedural adequacy
- route completeness
- evidence sufficiency

This would give TRM a much better supervision target than simple win/loss scoring.

## A Practical TRM Training Stack

If Claude wants a concrete, citable synthesis, this is the best current formulation:

### Stage A. Knowledge substrate first

Use K3D / PM-KR style procedural memory as the substrate.

Reason:
- TRM should not waste tiny capacity memorizing what can live in explicit procedural memory.

### Stage B. On-policy teacher-student rollout

Let TRM generate:
- recursive trajectory,
- intermediate states,
- final answer,
- confidence / uncertainty indicators.

Teacher evaluates the actual student rollout, not a static dataset only.

Reason:
- aligns with modern on-policy distillation logic,
- avoids train-test mismatch.

### Stage C. Multi-axis decomposition

Teacher decomposes:
- correct fragments
- incorrect fragments
- honesty signals
- missing honesty
- better/worse trajectory comparisons

Reason:
- turns each rollout into multi-signal supervision.

### Stage D. Hybrid objective

Use a hybrid loss:
- supervised / imitation term for final answer
- contrastive correctness term
- contrastive honesty term
- contrastive better/worse trajectory term
- route-level auxiliary loss
- optional reward-model distillation term

Reason:
- no single loss is sufficient.

### Stage E. Distill compactly, not blindly

Use teacher-assistant stepping when the gap is too large.

Reason:
- preserves learnability for a small recursive student.

## Why This Could Improve TRM More Than Plain Scaling

The reason this path is strategically attractive is that it exploits what TRM is already good at:
- recurrence,
- compactness,
- explicit iterative refinement.

Scaling alone would move TRM toward becoming just another small Transformer.  
Teacher-student + contrastive learning instead amplifies what makes TRM distinct:
- recursive update quality,
- calibration,
- structured state transitions,
- compact reasoning under constrained capacity.

In other words:

> the best way to improve TRM is probably not to make it larger first, but to make its recursive training signal denser, more structured, and more honest.

## Risks and Constraints

Claude should cite these carefully.

### 1. Over-distilling frontier biases

If the teacher is wrong, biased, or overconfident, TRM can inherit that behavior efficiently.

Mitigation:
- multi-teacher evaluation
- explicit uncertainty scoring
- reward-model families, not single reward proxies

### 2. Contrastive collapse from bad positives/negatives

Contrastive learning is only as good as the pair/triplet quality.

Mitigation:
- hard-negative curation
- route-aware negative mining
- memory-aware retrieval negatives

### 3. Tiny-model capacity ceiling

There is a real limit to how much a very small model can absorb from a much larger teacher.

Mitigation:
- staged distillation
- route-level specialization
- external procedural memory instead of full memorization

### 4. Scalar_Wizzard source gap

A second local repo was requested but was not locatable in the current workspace.

Mitigation:
- treat this report as version 1 grounded in `AI-RLWHF` plus external primary literature
- extend later if the missing repository path is provided

## Recommended Language For Claude

If Claude wants a compact thesis statement to reuse:

> AI-RLWHF already points toward the right training regime for TRM: not scalar reward alone, but teacher-student supervision decomposed into correctness, honesty, and cross-response contrast. In the broader literature, this aligns with the shift from static offline distillation toward on-policy, uncertainty-aware distillation for compact models. Applied to TRM, the key opportunity is recursive-trace distillation: supervising not only the final answer, but the quality, calibration, and geometry of the recursive reasoning path itself.

## Recommended Citations

For the teacher-student foundation:
- Hinton et al. 2015, "Distilling the Knowledge in a Neural Network"
- Wang et al. 2020, "MiniLM"
- Tan et al. 2023, "GKD"
- Song and Zheng 2026, "A Survey of On-Policy Distillation for Large Language Models"

For contrastive learning:
- Gao et al. 2021, "SimCSE"
- local `AI-RLWHF` contrastive-honesty specification and implementation report

For preference / reward distillation:
- Fisch et al. 2024, "Robust Preference Optimization through Reward Model Distillation"

For modern architecture context:
- Jiang et al. 2024, "Mixtral of Experts"
- DeepSeek-AI 2024, "DeepSeek-V2"
- He et al. 2024, "HMT"

For TRM itself:
- Jolicoeur-Martineau 2025, "Less is More: Recursive Reasoning with Tiny Networks"

## Bottom Line

The strongest synthesis available right now is:

- `AI-RLWHF` provides the **teacher-student honesty and decomposition logic**
- contrastive learning provides the **representation-shaping mechanism**
- TRM provides the **compact recursive learner**
- K3D provides the **external procedural memory substrate**

Together, these suggest a path where TRM improves not by becoming a scaled-down imitation of a frontier LLM, but by becoming a **better-calibrated, better-routed, recursively distilled compact reasoner operating over explicit memory**.
