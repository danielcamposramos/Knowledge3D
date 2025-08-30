Live Session Logs (Raw)

- Purpose: Raw JSONL logs and server outputs from local Live Mode runs, published for reproducibility and audit.
- Source: Originally produced under ../Knowledge3D.local/logs per DEVELOPMENT protocol.
- Note: This folder contains an explicit exception to the usual guidance to keep session logs out of the repo. Added per request to document recent runs.

Files
- session-20250828-184312.jsonl: Boot + hello logging smoke test.
- session-20250829-202337.jsonl: Short join/leave + command.
- session-20250829-202410.jsonl: Interactive navigation + exploration with model prediction sample.
- session-20250829-204826.jsonl: Extended mixed interactions.
- server.out: Live server stdout/stderr excerpt showing deprecation warnings and connection closures.
- server.pid: PID snapshot captured during a session.
- CHECKSUMS.txt: SHA-256 for all files in this folder.

Verification
- Compare SHA-256 hashes in CHECKSUMS.txt with those reported in prior summaries or external manifests.
- Logs follow the JSONL schema emitted by knowledge3d.bridge.live_server (presence, chat, chat_response, model_prediction, etc.).

Related Artifacts
- docs/reports/training/: Session summaries and derived task graphs built from these logs.
- docs/reports/models/: Trained intent classifier artifact (intent.pkl) with checksums.

Privacy and Scope
- Logs are system and test-user interactions only; no private data.
- Future runs will continue to write to ../Knowledge3D.local/logs by default; mirror here selectively as needed.

