# K3D Audio Architecture (Draft)

Goals:
- Low-latency, scalable voice for humans (spatial or non-spatial).
- Full-duplex speech IO for AI agents: streaming ASR + TTS.
- Open-source first; consider proprietary systems for inspiration.

## Options

- WebRTC (recommended):
  - Pros: Browser-native, Opus codec, echo cancellation, device selection, SFU support.
  - Stack: LiveKit (open core), mediasoup, Janus, Pion (Go), WebRTC.rs.
  - Usage: Viewer connects via WebRTC; agent endpoints run server-side.

- Mumble/Murmur (open-source TeamSpeak alternative):
  - Pros: Lightweight, Opus, stable.
  - Approach: Run a Murmur server; bridge chat/presence in K3D; optional 3D positional audio.

- TeamSpeak (proprietary):
  - Pros: Lightweight and battle-tested. SDK for plugins.
  - Approach: Optional bridge for organizations already invested in TS.

## AI Speech IO

- ASR (streaming):
  - Whisper (faster-whisper), Vosk, WebRTC VAD + chunked streaming.
  - Pipeline: mic → VAD → chunk → ASR → text → K3D intent.

- TTS (streaming):
  - Coqui TTS, Piper (lightweight), Mimic3; neural vocoders.
  - Pipeline: text → TTS (streamed chunks) → WebRTC/Opus.

- Closed options to study:
  - Microsoft VibeVoice (research) for expressive/controllable speech.
  - Commercial cloud TTS/ASR for benchmarks.

## Integration Plan

1) WebRTC signaling service (Node/Go) and SFU (mediasoup/LiveKit).
2) Viewer mic capture → WebRTC peer; room per K3D house (#channel alignment).
3) Agent endpoint (server) registers as a peer in the room.
4) ASR + NLU: map speech → `/goto`, `/annotate`, etc.; emit over WS bridge.
5) TTS: agent responses/audio status streamed back.

## Near-Term MVP

- Start with chat + WS commands (done), add `/join`, `/nick`, `/me`, `/msg` (done).
- Add WebRTC mic capture in viewer guarded by a toggle; route to noop server.
- Add minimal ASR server using faster-whisper to accept WebSocket PCM/Opus and emit text.
- Optional: Mumble bridge for drop-in voice rooms mapped to channels.

