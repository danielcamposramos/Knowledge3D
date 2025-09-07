RPN Runtime in K3D

Purpose
- Retain a precise, auditable policy core for navigation and interaction, complementing learned models. RPN is evaluated during inference to keep deterministic safety/logic constraints.

Where it lives
- `knowledge3d/core/rpn.py` — rule sets and evaluation machinery.
- Integrated by `knowledge3d/bridge/enhanced_chat_processor.py` and the live server for mapping intents/actions to concrete behaviors.

Interaction with learned models
- Intent model (HF): classifies user text into action types; RPN checks and refines action parameters.
- World model (RSSM): predicts spatial dynamics; RPN constraints ensure moves/orbits/goto adhere to policy.
- Gazetteer/aliases & spatial address normalize labels; RPN enforces disambiguation thresholds.

Logging & Training
- RPN emits decisions; logs record action rationales and outcomes. These logs fuel improved intent and world models while RPN remains the safety net.

Configuration
- Rules can be updated incrementally; see `knowledge3d/core/rpn.py` for patterns.

