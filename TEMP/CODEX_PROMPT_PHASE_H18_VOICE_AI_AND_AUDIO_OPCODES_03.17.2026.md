# Phase H18 — Live Voice AI Mode + Audio Processing Opcodes

**Depends on:** H16c (Multi-Modal Provider Layer), H17 (Universal Knowledge Foundation)
**Creates:**
- `knowledge3d/tools/voice_session.py` — Live voice interaction loop
- `knowledge3d/tools/audio_pipeline.py` — FFmpeg/Vibe audio preprocessing
- `viewer/src/rpn/audioOps.ts` — Browser-side audio RPN opcodes
**Tests:** `tests/test_voice_session.py`, `tests/test_audio_pipeline.py`, `viewer/tests/audioOps.test.ts`

---

## Vision

K3D is a living, always-on system. A natural way to interact with it is **voice** — speak a question, hear the answer. This phase adds:

1. **Live Voice AI Mode:** Continuous STT → Galaxy query → TTS response loop
2. **Audio Preprocessing Pipeline:** FFmpeg-based conversion + segmentation (Vibe pattern)
3. **Audio Processing RPN Opcodes:** Composable audio operations as part of K3D's RPN engine

---

## Part 1: Live Voice AI Mode

### Architecture

```
                    ┌──────────────────────────────┐
                    │   Live Voice Session Loop     │
                    │                               │
Microphone ──→ STT Provider ──→ Text Query          │
                    │               ↓               │
                    │        Galaxy Query Path       │
                    │        (same as benchmark)     │
                    │               ↓               │
                    │        Text Answer             │
                    │               ↓               │
                    │        TTS Provider ──→ Speaker│
                    │               ↓               │
                    │        health_log.jsonl        │
                    │        (sleep-time consolidation)
                    └──────────────────────────────┘
```

**Key:** Voice queries go through the SAME path as typed queries and benchmarks. The voice session is just an I/O adapter — the Galaxy doesn't know or care if the question came from keyboard, voice, or a benchmark file.

### VoiceSession Class

```python
class VoiceSession:
    """Live voice interaction loop — speak questions, hear answers."""

    def __init__(
        self,
        *,
        stt_provider: SpeechToTextProvider | None = None,
        tts_provider: TextToSpeechProvider | None = None,
        augmentation_provider: AugmentationProvider | None = None,
        language: str = "en",
        output_dir: str | None = None,
    ) -> None:
        self.stt = stt_provider or create_stt_provider()
        self.tts = tts_provider or create_tts_provider()
        self.augmentation = augmentation_provider or create_provider()
        self.language = language
        self.output_dir = Path(output_dir or tempfile.mkdtemp(prefix="k3d_voice_"))
        self.health_log = self.output_dir / "health_log.jsonl"
        self._running = False

    def transcribe_file(self, audio_path: str) -> TranscriptionResult:
        """Transcribe a single audio file to text."""
        # Preprocess if needed (non-WAV, video, multi-channel)
        processed = preprocess_audio(audio_path, output_dir=str(self.output_dir))
        return self.stt.transcribe(processed, language=self.language)

    def query_and_respond(self, question: str) -> str:
        """Send text query through augmentation path, return text answer.

        This is the same path benchmark questions take.
        Logs to health_log.jsonl for sleep-time consolidation.
        """
        result = self.augmentation.augment(question, {
            "name": "voice_query",
            "domain_hint": "General",
            "source": "voice",
        })
        answer = result.summary
        # Log for sleep-time consolidation
        self._log_interaction(question, answer, result)
        return answer

    def speak(self, text: str) -> str:
        """Convert text to speech, return path to audio file."""
        ts = int(time.time() * 1000)
        output_path = str(self.output_dir / f"response_{ts}.mp3")
        self.tts.synthesize(text, output_path=output_path)
        return output_path

    def process_audio_query(self, audio_path: str) -> tuple[str, str]:
        """Full pipeline: audio → transcribe → query → answer → speech.

        Returns (text_answer, audio_answer_path).
        """
        transcript = self.transcribe_file(audio_path)
        answer = self.query_and_respond(transcript.text)
        audio_path = self.speak(answer)
        return answer, audio_path

    def _log_interaction(self, question: str, answer: str, result: AugmentationResult) -> None:
        """Log to health_log.jsonl for sleep-time consolidation."""
        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "source": "voice",
            "question": question,
            "answer": answer,
            "domain": result.domain,
            "confidence": result.confidence,
            "provider": result.provider,
        }
        with open(self.health_log, "a") as f:
            f.write(json.dumps(entry) + "\n")
```

### Search Live Voice Mode

For continuous listening (future — requires microphone access):

```python
def start_voice_loop(self, *, listen_seconds: float = 10.0) -> None:
    """Start continuous listen → process → respond loop.

    Requires: system microphone access via arecord/parecord.
    """
    self._running = True
    while self._running:
        # Record audio chunk
        chunk_path = self._record_chunk(listen_seconds)
        if not chunk_path:
            continue
        # Process
        try:
            answer, audio = self.process_audio_query(chunk_path)
            # Play response
            self._play_audio(audio)
        except Exception as exc:
            print(f"Voice loop error: {exc}")
        # Cleanup temp chunk
        Path(chunk_path).unlink(missing_ok=True)

def stop(self) -> None:
    self._running = False

def _record_chunk(self, seconds: float) -> str | None:
    """Record from default microphone via arecord."""
    path = str(self.output_dir / f"chunk_{int(time.time())}.wav")
    try:
        subprocess.run(
            ["arecord", "-f", "S16_LE", "-r", "16000", "-c", "1",
             "-d", str(int(seconds)), path],
            check=True, capture_output=True, timeout=seconds + 5,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, subprocess.CalledProcessError):
        return None
    return path if Path(path).is_file() and Path(path).stat().st_size > 100 else None

def _play_audio(self, path: str) -> None:
    """Play audio file via aplay/paplay."""
    for player in ("paplay", "aplay", "ffplay"):
        try:
            subprocess.run([player, path], check=True, capture_output=True, timeout=30)
            return
        except (FileNotFoundError, subprocess.CalledProcessError):
            continue
```

---

## Part 2: Audio Preprocessing Pipeline

### Vibe Pattern (How It Works)

Vibe v3.0.11 on this machine:
- Binary: `/usr/bin/vibe` (ELF 64-bit, ~20MB)
- Lib: `/usr/lib/vibe/locales/` (18 language JSON packs only — NO bundled FFmpeg)
- Uses system FFmpeg: `/usr/bin/ffmpeg`
- Launch: `WEBKIT_DISABLE_COMPOSITING_MODE=1 /usr/bin/vibe`
- Server mode: `WEBKIT_DISABLE_COMPOSITING_MODE=1 /usr/bin/vibe --server` → port 3022

**Vibe's preprocessing command** (extracted from issue logs):
```bash
/usr/bin/ffmpeg -i [input] -ar 16000 -ac 1 -c:a pcm_s16le [output.wav]
```
This converts any audio/video to whisper-compatible format: 16kHz, mono, 16-bit PCM WAV.

### preprocess_audio() Function

```python
def preprocess_audio(
    input_path: str,
    *,
    output_dir: str | None = None,
    sample_rate: int = 16000,
    channels: int = 1,
    codec: str = "pcm_s16le",
) -> str:
    """Convert any audio/video to whisper-compatible WAV using system FFmpeg.

    Replicates the Vibe preprocessing pattern.
    If input is already 16kHz mono WAV, returns input unchanged.
    """
    inp = Path(input_path)
    if not inp.is_file():
        raise FileNotFoundError(f"Audio file not found: {input_path}")

    # Quick check: if already WAV, probe format
    if inp.suffix.lower() == ".wav" and _is_whisper_compatible(input_path):
        return input_path

    # Convert via FFmpeg
    out_dir = Path(output_dir or tempfile.mkdtemp(prefix="k3d_audio_"))
    out_path = out_dir / f"{inp.stem}_16k_mono.wav"

    ffmpeg_bin = _find_ffmpeg()
    cmd = [
        ffmpeg_bin, "-y", "-i", str(inp),
        "-ar", str(sample_rate),
        "-ac", str(channels),
        "-c:a", codec,
        str(out_path),
    ]
    result = subprocess.run(cmd, check=False, capture_output=True, text=True, timeout=120)
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg conversion failed: {result.stderr[:500]}")
    return str(out_path)


def _find_ffmpeg() -> str:
    """Locate FFmpeg binary. Check common paths."""
    for candidate in ("ffmpeg", "/usr/bin/ffmpeg", "/usr/local/bin/ffmpeg"):
        try:
            result = subprocess.run([candidate, "-version"],
                                     check=False, capture_output=True, timeout=5)
            if result.returncode == 0:
                return candidate
        except FileNotFoundError:
            continue
    raise FileNotFoundError("FFmpeg not found. Install with: apt install ffmpeg")


def _is_whisper_compatible(path: str) -> bool:
    """Check if WAV file is already 16kHz mono 16-bit PCM."""
    try:
        import wave
        with wave.open(path, "rb") as wf:
            return (wf.getframerate() == 16000 and
                    wf.getnchannels() == 1 and
                    wf.getsampwidth() == 2)
    except Exception:
        return False
```

### Extended Pipeline Operations

```python
def extract_audio_from_video(video_path: str, *, output_dir: str | None = None) -> str:
    """Extract audio track from video container."""
    inp = Path(video_path)
    out_dir = Path(output_dir or tempfile.mkdtemp(prefix="k3d_audio_"))
    out_path = out_dir / f"{inp.stem}_audio.wav"
    cmd = [_find_ffmpeg(), "-y", "-i", str(inp), "-vn",
           "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", str(out_path)]
    subprocess.run(cmd, check=True, capture_output=True, timeout=300)
    return str(out_path)


def split_by_silence(audio_path: str, *, min_silence_ms: int = 500,
                      silence_thresh_db: int = -40, output_dir: str | None = None) -> list[str]:
    """Split audio at silence boundaries using FFmpeg silencedetect filter.

    Returns list of segment file paths.
    """
    # Step 1: Detect silence boundaries
    ffmpeg_bin = _find_ffmpeg()
    detect_cmd = [
        ffmpeg_bin, "-i", audio_path,
        "-af", f"silencedetect=noise={silence_thresh_db}dB:d={min_silence_ms/1000}",
        "-f", "null", "-",
    ]
    result = subprocess.run(detect_cmd, check=False, capture_output=True, text=True, timeout=120)

    # Step 2: Parse silence boundaries from stderr
    import re
    boundaries = []
    for match in re.finditer(r"silence_end: ([\d.]+)", result.stderr):
        boundaries.append(float(match.group(1)))

    # Step 3: Split at boundaries
    out_dir = Path(output_dir or tempfile.mkdtemp(prefix="k3d_segments_"))
    segments = []
    prev = 0.0
    for i, boundary in enumerate(boundaries):
        seg_path = out_dir / f"segment_{i:04d}.wav"
        split_cmd = [
            ffmpeg_bin, "-y", "-i", audio_path,
            "-ss", str(prev), "-to", str(boundary),
            "-c:a", "pcm_s16le", str(seg_path),
        ]
        subprocess.run(split_cmd, check=False, capture_output=True, timeout=60)
        if seg_path.is_file() and seg_path.stat().st_size > 100:
            segments.append(str(seg_path))
        prev = boundary

    # Final segment (after last silence boundary)
    if prev > 0:
        seg_path = out_dir / f"segment_{len(boundaries):04d}.wav"
        split_cmd = [ffmpeg_bin, "-y", "-i", audio_path,
                     "-ss", str(prev), "-c:a", "pcm_s16le", str(seg_path)]
        subprocess.run(split_cmd, check=False, capture_output=True, timeout=60)
        if seg_path.is_file() and seg_path.stat().st_size > 100:
            segments.append(str(seg_path))

    return segments


def get_audio_metadata(audio_path: str) -> dict[str, Any]:
    """Probe audio file metadata using FFmpeg."""
    ffmpeg_bin = _find_ffmpeg()
    cmd = [ffmpeg_bin.replace("ffmpeg", "ffprobe"), "-v", "quiet",
           "-print_format", "json", "-show_format", "-show_streams", str(audio_path)]
    result = subprocess.run(cmd, check=False, capture_output=True, text=True, timeout=30)
    if result.returncode != 0:
        return {}
    return json.loads(result.stdout) if result.stdout else {}
```

---

## Part 3: Audio Processing RPN Opcodes (Browser-Side)

### Rationale

K3D's RPN engine already has `meshOps`, `pathOps`, `mat4Ops`, `domOps`. Audio is the next modality. These opcodes let K3D compose audio processing programs in the same way it composes visual and DOM programs.

### File: `viewer/src/rpn/audioOps.ts`

Map FFmpeg's discrete operations to RPN opcodes:

```typescript
/**
 * Audio RPN Opcodes — composable audio processing for K3D.
 *
 * These operate on AudioBuffer objects on the RPN stack.
 * Browser-side uses Web Audio API (AudioContext, OfflineAudioContext).
 */

export const AUDIO_OPS: Record<string, (ctx: RpnContext) => void> = {
  // === Container / Transport ===

  /** AUDIO_LOAD: (url: string) → (AudioBuffer) — fetch and decode audio file */
  AUDIO_LOAD: async (ctx) => { /* fetch + decodeAudioData */ },

  /** AUDIO_PROBE: (AudioBuffer) → (metadata: {duration, sampleRate, channels}) */
  AUDIO_PROBE: (ctx) => {
    const buf = ctx.stack.pop() as AudioBuffer;
    ctx.stack.push({
      duration: buf.duration,
      sampleRate: buf.sampleRate,
      channels: buf.numberOfChannels,
      length: buf.length,
    });
  },

  // === Signal Processing ===

  /** AUDIO_RESAMPLE: (AudioBuffer, targetRate: number) → (AudioBuffer) */
  AUDIO_RESAMPLE: (ctx) => { /* OfflineAudioContext at target rate */ },

  /** AUDIO_MONO: (AudioBuffer) → (AudioBuffer) — mix down to single channel */
  AUDIO_MONO: (ctx) => { /* Average all channels */ },

  /** AUDIO_GAIN: (AudioBuffer, gainDb: number) → (AudioBuffer) */
  AUDIO_GAIN: (ctx) => { /* GainNode processing */ },

  /** AUDIO_NORMALIZE: (AudioBuffer) → (AudioBuffer) — normalize to peak 0dB */
  AUDIO_NORMALIZE: (ctx) => { /* Find peak, apply inverse gain */ },

  /** AUDIO_TRIM: (AudioBuffer, startSec: number, endSec: number) → (AudioBuffer) */
  AUDIO_TRIM: (ctx) => { /* Slice Float32Array by sample offsets */ },

  /** AUDIO_CONCAT: (AudioBuffer, AudioBuffer) → (AudioBuffer) — join sequentially */
  AUDIO_CONCAT: (ctx) => { /* Create new buffer, copy both */ },

  /** AUDIO_FADE_IN: (AudioBuffer, durationSec: number) → (AudioBuffer) */
  AUDIO_FADE_IN: (ctx) => { /* Linear ramp from 0 to 1 */ },

  /** AUDIO_FADE_OUT: (AudioBuffer, durationSec: number) → (AudioBuffer) */
  AUDIO_FADE_OUT: (ctx) => { /* Linear ramp from 1 to 0 */ },

  /** AUDIO_MIX: (AudioBuffer, AudioBuffer, ratio: number) → (AudioBuffer) */
  AUDIO_MIX: (ctx) => { /* Weighted sum of two buffers */ },

  // === Analysis ===

  /** AUDIO_PEAK: (AudioBuffer) → (peakAmplitude: number) */
  AUDIO_PEAK: (ctx) => {
    const buf = ctx.stack.pop() as AudioBuffer;
    const data = buf.getChannelData(0);
    let peak = 0;
    for (let i = 0; i < data.length; i++) {
      const abs = Math.abs(data[i]);
      if (abs > peak) peak = abs;
    }
    ctx.stack.push(peak);
  },

  /** AUDIO_RMS: (AudioBuffer) → (rmsEnergy: number) */
  AUDIO_RMS: (ctx) => {
    const buf = ctx.stack.pop() as AudioBuffer;
    const data = buf.getChannelData(0);
    let sum = 0;
    for (let i = 0; i < data.length; i++) sum += data[i] * data[i];
    ctx.stack.push(Math.sqrt(sum / data.length));
  },

  /** AUDIO_DURATION: (AudioBuffer) → (seconds: number) */
  AUDIO_DURATION: (ctx) => {
    const buf = ctx.stack.pop() as AudioBuffer;
    ctx.stack.push(buf.duration);
  },

  // === Playback (browser-side only) ===

  /** AUDIO_PLAY: (AudioBuffer) → () — play through speakers */
  AUDIO_PLAY: (ctx) => { /* AudioContext.createBufferSource().start() */ },

  /** AUDIO_STOP: () → () — stop current playback */
  AUDIO_STOP: (ctx) => { /* source.stop() */ },
};
```

### RPN Engine Integration

Register `audioOps` alongside existing op sets:

```typescript
// In rpn/index.ts
export function createAudioRpnEngine(): RpnEngine {
  return new RpnEngine({ ...AUDIO_OPS });
}

export function createFullRpnEngine(): RpnEngine {
  return new RpnEngine({
    ...meshOps, ...pathOps, ...mat4Ops, ...domOps, ...AUDIO_OPS,
  });
}
```

### Example RPN Programs

```rpn
# Prepare audio file for whisper transcription
"https://example.com/lecture.mp3" AUDIO_LOAD
16000 AUDIO_RESAMPLE
AUDIO_MONO
AUDIO_NORMALIZE

# Analyze audio clip
"clip.wav" AUDIO_LOAD
AUDIO_PROBE      # → {duration, sampleRate, channels}
AUDIO_PEAK       # → peak amplitude
AUDIO_RMS        # → RMS energy

# Create fade-in/out transition
"intro.wav" AUDIO_LOAD
2.0 AUDIO_FADE_IN
"outro.wav" AUDIO_LOAD
3.0 AUDIO_FADE_OUT
AUDIO_CONCAT
```

---

## Part 4: FFmpeg as K3D Internal Tool (Python Side)

### Audio Opcodes on Python Side (Ingestion Path)

Mirror the browser-side opcodes for server-side processing via FFmpeg subprocess:

```python
# In audio_pipeline.py

AUDIO_OPERATIONS = {
    "AUDIO_RESAMPLE": lambda inp, rate: _ffmpeg_convert(inp, ar=rate),
    "AUDIO_MONO": lambda inp: _ffmpeg_convert(inp, ac=1),
    "AUDIO_NORMALIZE": lambda inp: _ffmpeg_filter(inp, "loudnorm"),
    "AUDIO_TRIM": lambda inp, start, end: _ffmpeg_trim(inp, start, end),
    "AUDIO_CONCAT": lambda *inputs: _ffmpeg_concat(inputs),
    "AUDIO_GAIN": lambda inp, db: _ffmpeg_filter(inp, f"volume={db}dB"),
    "AUDIO_FADE_IN": lambda inp, dur: _ffmpeg_filter(inp, f"afade=t=in:d={dur}"),
    "AUDIO_FADE_OUT": lambda inp, dur: _ffmpeg_filter(inp, f"afade=t=out:d={dur}"),
    "AUDIO_PROBE": lambda inp: get_audio_metadata(inp),
    "AUDIO_SPLIT_SILENCE": lambda inp, **kw: split_by_silence(inp, **kw),
}


def execute_audio_pipeline(input_path: str, operations: list[tuple[str, ...]]) -> str:
    """Execute a sequence of audio operations.

    Example:
        execute_audio_pipeline("lecture.mp4", [
            ("AUDIO_RESAMPLE", 16000),
            ("AUDIO_MONO",),
            ("AUDIO_NORMALIZE",),
        ])
    """
    current = input_path
    for op_tuple in operations:
        op_name = op_tuple[0]
        args = op_tuple[1:]
        fn = AUDIO_OPERATIONS.get(op_name)
        if fn is None:
            raise ValueError(f"Unknown audio operation: {op_name}")
        current = fn(current, *args)
    return current
```

---

## Tests

### test_audio_pipeline.py

```python
def test_find_ffmpeg():
    """FFmpeg binary is discoverable on this system."""
    path = _find_ffmpeg()
    assert "ffmpeg" in path

def test_preprocess_audio_wav_passthrough(tmp_path):
    """16kHz mono WAV passes through unchanged."""
    # Create a valid 16kHz mono WAV
    wav_path = _create_test_wav(tmp_path / "test.wav", rate=16000, channels=1)
    result = preprocess_audio(str(wav_path))
    assert result == str(wav_path)  # No conversion needed

def test_preprocess_audio_converts_stereo(tmp_path):
    """Stereo WAV gets converted to mono."""
    wav_path = _create_test_wav(tmp_path / "stereo.wav", rate=44100, channels=2)
    result = preprocess_audio(str(wav_path), output_dir=str(tmp_path))
    assert Path(result).is_file()
    assert result != str(wav_path)  # Different file

def test_get_audio_metadata(tmp_path):
    """Probe returns duration and format info."""
    wav_path = _create_test_wav(tmp_path / "test.wav")
    meta = get_audio_metadata(str(wav_path))
    assert "format" in meta or "streams" in meta

def test_split_by_silence_empty_returns_original(tmp_path):
    """Audio with no silence returns single segment or empty list."""
    wav_path = _create_test_wav(tmp_path / "continuous.wav")
    segments = split_by_silence(str(wav_path), output_dir=str(tmp_path / "segs"))
    assert isinstance(segments, list)
```

### test_voice_session.py

```python
def test_voice_session_creates(tmp_path):
    """VoiceSession instantiates with default providers."""
    session = VoiceSession(output_dir=str(tmp_path))
    assert session.output_dir == tmp_path

def test_voice_session_logs_interaction(tmp_path):
    """query_and_respond logs to health_log.jsonl."""
    # Mock augmentation provider
    session = VoiceSession(
        augmentation_provider=MockAugProvider(),
        output_dir=str(tmp_path),
    )
    answer = session.query_and_respond("What is water?")
    assert answer  # Got some answer
    assert session.health_log.is_file()
    with open(session.health_log) as f:
        entry = json.loads(f.readline())
    assert entry["source"] == "voice"
    assert entry["question"] == "What is water?"
```

### viewer/tests/audioOps.test.ts

```typescript
describe('audioOps', () => {
  test('AUDIO_PROBE pushes metadata', () => {
    // Mock AudioBuffer
    const buf = { duration: 5.0, sampleRate: 16000, numberOfChannels: 1, length: 80000 };
    const ctx = createTestContext([buf]);
    AUDIO_OPS.AUDIO_PROBE(ctx);
    expect(ctx.stack.pop()).toEqual({
      duration: 5.0, sampleRate: 16000, channels: 1, length: 80000,
    });
  });

  test('AUDIO_PEAK finds maximum amplitude', () => {
    const data = new Float32Array([0.1, -0.5, 0.3, -0.8, 0.2]);
    const buf = { getChannelData: () => data, length: 5 };
    const ctx = createTestContext([buf]);
    AUDIO_OPS.AUDIO_PEAK(ctx);
    expect(ctx.stack.pop()).toBeCloseTo(0.8);
  });

  test('AUDIO_RMS computes root mean square', () => {
    const data = new Float32Array([1, 1, 1, 1]);
    const buf = { getChannelData: () => data, length: 4 };
    const ctx = createTestContext([buf]);
    AUDIO_OPS.AUDIO_RMS(ctx);
    expect(ctx.stack.pop()).toBeCloseTo(1.0);
  });

  test('AUDIO_DURATION pushes seconds', () => {
    const buf = { duration: 12.5 };
    const ctx = createTestContext([buf]);
    AUDIO_OPS.AUDIO_DURATION(ctx);
    expect(ctx.stack.pop()).toBe(12.5);
  });
});
```

---

## File Changes Summary

| File | Action |
|------|--------|
| `knowledge3d/tools/voice_session.py` | **NEW** — VoiceSession class, voice loop, health logging |
| `knowledge3d/tools/audio_pipeline.py` | **NEW** — FFmpeg preprocessing, silence split, metadata probe |
| `viewer/src/rpn/audioOps.ts` | **NEW** — 16 browser-side audio RPN opcodes |
| `viewer/src/rpn/index.ts` | **MODIFY** — Register audioOps, add createAudioRpnEngine() |
| `tests/test_voice_session.py` | **NEW** |
| `tests/test_audio_pipeline.py` | **NEW** |
| `viewer/tests/audioOps.test.ts` | **NEW** |

---

## Vibe Integration Summary

| Component | How It Integrates |
|-----------|-------------------|
| Vibe GUI app | `/usr/bin/vibe` — user's standalone transcription tool |
| Vibe --server mode | `VibeServerProvider` (H16c) — OpenAI-compatible API on port 3022 |
| Vibe's FFmpeg pattern | `preprocess_audio()` replicates the same `-ar 16000 -ac 1 -c:a pcm_s16le` command |
| Vibe's whisper.cpp | K3D uses `faster-whisper` (Python-native) OR Vibe server (HTTP) |
| System FFmpeg | `audio_pipeline.py` uses system `/usr/bin/ffmpeg` for all conversion |
| Vibe models path | Not shared — K3D manages its own models via faster-whisper download |

---

## Success Criteria

1. `VoiceSession` instantiates and processes audio queries end-to-end
2. Voice queries logged to `health_log.jsonl` (same format as benchmark queries)
3. `preprocess_audio()` converts any audio/video to 16kHz mono WAV via FFmpeg
4. `split_by_silence()` segments audio at silence boundaries
5. `get_audio_metadata()` returns duration/format/codec info
6. Browser-side `audioOps` registered in RPN engine
7. `AUDIO_PROBE`, `AUDIO_PEAK`, `AUDIO_RMS`, `AUDIO_DURATION` pass unit tests
8. All existing tests still pass (non-regression)
9. FFmpeg binary detected or clear error message if missing

---

## Connection to Broader Architecture

**Voice is just another I/O adapter:**
```
Voice Query → STT → Text → [same Galaxy path as everything] → Text → TTS → Audio
Typed Query → Text → [same Galaxy path as everything] → Text → Display
Benchmark Q → Text → [same Galaxy path as everything] → Text → health_log
```

**Audio opcodes complete the modality ring:**
```
Visual: meshOps + pathOps + mat4Ops  (3D geometry)
DOM:    domOps                        (web projection)
Audio:  audioOps                      (sound processing)
```

All three modalities compose in the same RPN engine, enabling cross-modal programs:
```rpn
# Load 3D object, play its associated audio, project its description to DOM
"shelf_book_1" MESH_LOAD
"shelf_book_1_audio" AUDIO_LOAD AUDIO_PLAY
"shelf_book_1" DOM_P DOM_TEXT DOM_EMIT
```
