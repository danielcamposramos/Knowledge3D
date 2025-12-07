# Unified Signal Specification — Frequency-Time Architecture

**Version**: 1.0
**Status**: Implementation Ready
**License**: CC-BY-4.0 (Documentation), Apache 2.0 (Implementation)
**Date**: December 2025

---

## Abstract

This specification defines K3D's **Unified Signal Architecture** — a single framework for representing all time-varying signals (audio, radio, video, sensor data) as frequency components over time. It formalizes:

- The **Frequency-Time Bridge** connecting audio, images, and radio
- **Spectrogram as Procedural Image** — audio visualization using VectorDotMap
- **SDR (Software Defined Radio)** integration for radio frequency signals
- **Binaural Spatial Audio** for 3D sound in Galaxy/House environments
- **Bidirectional conversion**: Audio ↔ Image (spectrogram ↔ sonification)

This architecture embodies the core insight: **All signals are vibration in frequency over time**. The same mathematical representation (STFT → spectrogram → VectorDotMap) applies to 20 Hz audio and 2.4 GHz WiFi.

---

## 1. Core Principle: Everything is Frequency Over Time

### 1.1 The Unifying Insight

| Signal Type | Frequency Range | Time Resolution | Same Math! |
|-------------|-----------------|-----------------|------------|
| **Audio (speech)** | 20 Hz - 20 kHz | ~50µs | STFT → Spectrogram |
| **Audio (ultrasound)** | 20 kHz - 200 kHz | ~5µs | STFT → Spectrogram |
| **Radio (FM)** | 88 MHz - 108 MHz | ~1µs | STFT → Waterfall |
| **Radio (WiFi)** | 2.4 GHz / 5 GHz | ~1ns | STFT → Spectrogram |
| **Video (motion)** | 24-120 Hz (fps) | 8-42ms | Frame difference → Temporal spectrum |
| **Sensor (vibration)** | 0.1 Hz - 10 kHz | ~100µs | STFT → Spectrogram |
| **Biological (EEG)** | 0.5 Hz - 100 Hz | ~10ms | STFT → Spectrogram |

**All are**: Time-series of frequency components, representable as 2D images (frequency × time).

### 1.2 The Unified Pipeline

```
ANY TIME-SERIES SIGNAL
        ↓
   [Windowing]           ← Hann, Hamming, Blackman
        ↓
   [FFT/STFT]            ← Same transform for all signals
        ↓
   [Frequency Scaling]   ← Mel (audio), Linear (radio), Log (vibration)
        ↓
   Spectrogram (Frequency × Time)
        ↓
   [VectorDotMap Encode] ← Procedural image representation
        ↓
   Field Coefficients (~2KB)
        ↓
   [Cross-Modal Bridge]  ← Links to Text Galaxy, other modalities
```

---

## 2. Audio Galaxy Architecture

### 2.1 Existing Sovereign Audio Codec

**File**: `knowledge3d/cranium/codecs/sovereign_ternary_audio_codec.py`

Current capabilities:
- MDCT/IMDCT transforms (GPU-native)
- Ternary quantization
- RPN integration via ModularRPNEngine

### 2.2 Enhanced Audio Pipeline

```rpn
# Audio encoding (waveform → procedural)
AUDIO_LOAD samples sample_rate    # Load raw samples
WINDOW_HANN frame_size            # Apply window function
STFT n_fft hop_length             # Short-Time Fourier Transform
MEL_SCALE n_mels fmin fmax        # Mel frequency scaling
DB_SCALE ref_db                   # Convert to decibels
VECTORDOTMAP_ENCODE               # Encode as procedural image
TERNARY_QUANT threshold           # Ternary quantization
GALAXY_STORE clip_id              # Store in Audio Galaxy

# Audio decoding (procedural → waveform)
GALAXY_LOAD clip_id
TERNARY_DEQUANT
VECTORDOTMAP_DECODE width height  # Reconstruct spectrogram
MEL_INVERSE                       # Mel to linear frequency
GRIFFIN_LIM iterations            # Phase reconstruction
ISTFT                             # Inverse STFT
AUDIO_RENDER                      # Output waveform
```

### 2.3 RPN Opcodes for Audio

```rpn
# Waveform operations
AUDIO_LOAD path                   # Load audio file
AUDIO_SAMPLES samples rate        # Create from array
AUDIO_DURATION                    # Get duration in seconds
AUDIO_RESAMPLE target_rate        # Resample to new rate

# Spectral analysis
WINDOW_HANN / WINDOW_HAMMING / WINDOW_BLACKMAN size
STFT n_fft hop_length             # Forward STFT
ISTFT                             # Inverse STFT
FFT / IFFT                        # Single-frame FFT

# Frequency scaling
MEL_SCALE n_mels fmin fmax        # Mel scale (perceptual, audio)
LINEAR_SCALE n_bins fmin fmax     # Linear scale (physical, radio)
LOG_SCALE n_bins fmin fmax        # Logarithmic scale (vibration)
BARK_SCALE n_bands                # Bark scale (psychoacoustic)

# Amplitude scaling
DB_SCALE ref                      # Power to decibels
DB_INVERSE                        # Decibels to power
NORMALIZE peak                    # Normalize amplitude

# Phase
PHASE_EXTRACT                     # Get phase from complex STFT
PHASE_VOCODER speed_ratio         # Time stretching
GRIFFIN_LIM iterations            # Phase reconstruction
```

---

## 3. Spectrogram as Procedural Image

### 3.1 The Bridge: Audio ↔ Image

A spectrogram IS an image:
- **X-axis**: Time
- **Y-axis**: Frequency
- **Color/brightness**: Amplitude

This means spectrograms can use the **same VectorDotMap codec** as regular images:

```rpn
# Audio → Procedural Image
AUDIO_LOAD speech.wav
SPECTROGRAM_GENERATE              # STFT + Mel + dB
VECTORDOTMAP_ENCODE               # Same encoder as images!
STORE "speech_visual"

# Procedural Image → Render at any resolution
LOAD "speech_visual"
VECTORDOTMAP_DECODE 1920 1080     # Render at 1080p
IMAGE_DISPLAY

# Or render at 4K from same data
VECTORDOTMAP_DECODE 3840 2160     # Same coefficients, higher detail
```

### 3.2 Benefits

1. **Unified storage**: Audio and images share the same format
2. **Cross-modal search**: Find similar spectrograms by image similarity
3. **Infinite resolution**: Render spectrogram at any zoom level
4. **Compression**: ~2KB per audio segment regardless of duration

### 3.3 PTX Kernel: Spectrogram Generation

```cuda
// spectrogram_to_vectordotmap.cu

__global__ void audio_to_spectrogram(
    const float* audio_samples,
    float* spectrogram,
    const float* window,
    const float* mel_filterbank,
    int n_samples,
    int n_fft,
    int hop_length,
    int n_mels,
    float sample_rate
) {
    // Standard STFT implementation
    // Output: (n_mels × n_frames) spectrogram
}

__global__ void spectrogram_to_field(
    const float* spectrogram,
    float* field_coefficients,
    int width,
    int height,
    int n_coefficients
) {
    // Fit VectorDotMap field to spectrogram
    // Same algorithm as image encoding
}
```

---

## 4. SDR (Software Defined Radio) Integration

### 4.1 Radio Signals = Audio at Higher Frequencies

The **exact same pipeline** works for radio:

| Parameter | Audio | Radio (WiFi) |
|-----------|-------|--------------|
| Sample rate | 44.1 kHz | 20 MHz |
| Frequency range | 20 Hz - 20 kHz | 2.4 GHz ± 10 MHz |
| FFT size | 2048 | 4096 |
| Representation | Spectrogram | Waterfall |

### 4.2 I/Q Sample Processing

Radio signals are captured as I/Q (In-phase/Quadrature) samples:

```rpn
# SDR signal processing
IQ_LOAD i_samples q_samples sample_rate center_freq
IQ_TO_COMPLEX                     # Combine I + jQ
STFT n_fft hop_length             # Same STFT as audio!
POWER_SPECTRUM                    # |FFT|²
DB_SCALE 10                       # Convert to dB
WATERFALL_RENDER                  # Render as spectrogram image
VECTORDOTMAP_ENCODE               # Store procedurally
```

### 4.3 SDR-Specific Opcodes

```rpn
# I/Q operations
IQ_LOAD i q rate center           # Load I/Q samples
IQ_TO_COMPLEX                     # Create complex signal
IQ_DEMODULATE mode                # AM/FM/SSB demodulation

# Frequency operations
FREQ_SHIFT offset                 # Shift center frequency
FREQ_FILTER low high              # Bandpass filter
DECIMATION factor                 # Reduce sample rate

# Modulation detection
MOD_DETECT                        # Auto-detect modulation type
CONSTELLATION_RENDER              # I/Q constellation diagram
WATERFALL_RENDER                  # Frequency-time waterfall

# Signal analysis
SIGNAL_DETECT threshold           # Detect signal presence
BANDWIDTH_ESTIMATE                # Estimate signal bandwidth
SNR_COMPUTE                       # Signal-to-noise ratio
```

### 4.4 Use Cases

1. **WiFi presence detection**: Spectrogram shows 2.4/5 GHz carriers
2. **Bluetooth tracking**: Frequency hopping patterns visible
3. **Interference analysis**: Overlapping signals in waterfall
4. **RF fingerprinting**: Unique signal signatures per device

---

## 5. Binaural Spatial Audio

### 5.1 3D Sound in Galaxy/House

K3D's spatial architecture (Galaxy = active memory, House = persistent storage) naturally supports 3D audio positioning:

```
┌─────────────────────────────────────────────┐
│              GALAXY SPACE                   │
│                                             │
│     [Sound Source A]                        │
│          ↓ 3D position                      │
│     (x=2.5, y=1.0, z=-3.0)                 │
│          ↓                                  │
│     [HRTF Processing]                       │
│          ↓                                  │
│     Left ear    Right ear                   │
│     delay/gain  delay/gain                  │
│          ↓                                  │
│     [Listener Position]                     │
│     (x=0, y=1.7, z=0)                      │
│                                             │
└─────────────────────────────────────────────┘
```

### 5.2 HRTF (Head-Related Transfer Function)

HRTF encodes how sound changes based on direction:

```rpn
# Spatial audio positioning
SOURCE_CREATE source_id
SOURCE_POSITION x y z             # 3D position in Galaxy coords
SOURCE_AUDIO audio_ref            # Reference to Audio Galaxy

# HRTF application
LISTENER_POSITION x y z           # Listener location
LISTENER_ORIENTATION yaw pitch roll
HRTF_COMPUTE source_id            # Compute binaural filters
ITD_APPLY                         # Interaural Time Difference
ILD_APPLY                         # Interaural Level Difference
BINAURAL_RENDER left right        # Output stereo

# Room acoustics
ROOM_SIZE width height depth
ROOM_MATERIALS walls ceiling floor
REVERB_COMPUTE                    # Early reflections + late reverb
OCCLUSION_COMPUTE obstacles       # Sound through/around objects
```

### 5.3 RPN Opcodes for Spatial Audio

```rpn
# Source management
SOURCE_CREATE id                  # Create audio source
SOURCE_POSITION x y z             # Set 3D position
SOURCE_VELOCITY vx vy vz          # For Doppler effect
SOURCE_DIRECTION dx dy dz         # Directivity vector
SOURCE_CONE inner outer           # Directional cone (degrees)

# Attenuation models
ATTEN_LINEAR ref_dist max_dist
ATTEN_INVERSE ref_dist rolloff
ATTEN_EXPONENTIAL ref_dist rolloff

# HRTF processing
HRTF_LOAD profile                 # Load HRTF dataset
HRTF_COMPUTE azimuth elevation    # Get binaural filters
ITD_COMPUTE distance angle        # Time difference (µs)
ILD_COMPUTE frequency angle       # Level difference (dB)

# Room simulation
REVERB_ROOM size rt60 damping     # Room reverb
EARLY_REFLECT surfaces count      # Early reflections
LATE_REVERB density diffusion     # Diffuse reverb tail
OCCLUSION material thickness      # Sound through walls

# Output
BINAURAL_MIX sources              # Mix all sources binaurally
AMBISONIC_ENCODE order            # Ambisonics encoding
SPEAKER_DECODE layout             # Decode to speaker array
```

### 5.4 Integration with House/Galaxy

```rpn
# Sound in House rooms
HOUSE_ROOM_ENTER "living_room"
ROOM_ACOUSTICS_LOAD               # Load room acoustic properties
SOURCE_POSITION_RELATIVE          # Position relative to room

# Sound in Galaxy space
GALAXY_POSITION x y z             # Absolute Galaxy coordinates
SPATIAL_ANCHOR node_id            # Attach sound to knowledge node
PROXIMITY_TRIGGER distance        # Activate when listener approaches
```

---

## 6. Bidirectional Audio ↔ Image

### 6.1 Spectrogram (Audio → Image)

Already covered in Section 3. Key point: **same VectorDotMap codec**.

### 6.2 Sonification (Image → Audio)

Convert images to sound by treating them as spectrograms:

```rpn
# Image → Audio (sonification)
IMAGE_LOAD picture.png
IMAGE_TO_GRAYSCALE                # Convert to single channel
IMAGE_RESIZE 128 duration_frames  # Height=frequency bins, Width=time

# Interpret as spectrogram
SPECTROGRAM_INTERPRET             # Treat image as spectrogram
DB_INVERSE                        # Convert brightness to amplitude
MEL_INVERSE                       # Mel to linear frequency
GRIFFIN_LIM 32                    # Reconstruct phase
ISTFT                             # Inverse STFT
AUDIO_NORMALIZE
AUDIO_PLAY
```

### 6.3 Creative Applications

```rpn
# Sonify a photograph
IMAGE_LOAD landscape.jpg
EDGE_DETECT                       # Extract edges
SONIFY                            # Convert to audio
# Result: High frequencies where edges are, silence in flat areas

# Sonify a drawing
DRAWING_LOAD sketch.vdm
VECTORDOTMAP_DECODE 128 256       # 128 freq bins, 256 time frames
SONIFY
# Result: Sound follows the drawn strokes

# Audio visualization feedback loop
AUDIO_LOAD music.wav
SPECTROGRAM_GENERATE
ARTISTIC_COLORMAP plasma          # Apply artistic colormap
FILTER_STYLIZE                    # Artistic filters
SONIFY                            # Convert modified image back to audio
# Result: Processed audio that sounds like its own visualization
```

---

## 7. Video as Signal

### 7.1 Video = Images Over Time

Video frames are a signal in the temporal frequency domain:

```
Video Signal:
- Sample rate: Framerate (24, 30, 60 fps)
- Amplitude: Pixel values
- Frequency: Temporal changes (motion)

Temporal Spectrum:
- DC component: Static background
- Low frequencies: Slow motion
- High frequencies: Fast motion, flicker
```

### 7.2 Temporal Analysis

```rpn
# Video temporal spectrum
VIDEO_LOAD clip.mp4
FRAMES_TO_TEMPORAL                # Stack frames along time axis
TEMPORAL_FFT                      # FFT along time dimension
MOTION_SPECTRUM_RENDER            # Visualize temporal frequencies

# Motion-based compression
MOTION_DETECT threshold
KEYFRAME_EXTRACT                  # Extract significant changes
DELTA_ENCODE                      # Encode differences
VECTORDOTMAP_TEMPORAL             # Procedural video encoding
```

### 7.3 Audio-Video Synchronization

```rpn
# Synchronized encoding
VIDEO_LOAD video.mp4
AUDIO_EXTRACT                     # Extract audio track
AUDIO_SPECTROGRAM                 # Audio → procedural image
VIDEO_PROCEDURAL                  # Video → procedural sequence
SYNC_TIMELINE audio video         # Align timelines
MULTIMODAL_STORE clip_id          # Store both with sync metadata
```

---

## 8. Cross-Modal Discovery

### 8.1 Shared Frequency Patterns

Because audio and images share the same representation, we can discover connections:

```rpn
# Find images that "look like" an audio clip
AUDIO_LOAD speech.wav
SPECTROGRAM_ENCODE                # Audio → VectorDotMap
SIMILARITY_SEARCH "image_galaxy" 0.8  # Find similar images
# Result: Images with similar frequency structure

# Find audio that "sounds like" an image
IMAGE_LOAD pattern.png
VECTORDOTMAP_ENCODE
SIMILARITY_SEARCH "audio_galaxy" 0.8  # Find similar spectrograms
# Result: Audio clips with matching spectral patterns
```

### 8.2 Emergent Cross-Modal Links

The Discovery Layer can identify:
- **Visual patterns** that correspond to **specific sounds**
- **Audio signatures** that match **visual textures**
- **Rhythmic patterns** shared between **music and motion**

---

## 9. Implementation

### 9.1 Files to Create/Modify

| File | Purpose | Status |
|------|---------|--------|
| `knowledge3d/cranium/audio_galaxy.py` | Audio storage/retrieval | Enhance |
| `knowledge3d/cranium/sdr_processor.py` | SDR signal processing | Create |
| `knowledge3d/cranium/binaural_audio.py` | Spatial audio | Create |
| `knowledge3d/cranium/sonification.py` | Image → Audio | Create |
| `knowledge3d/cranium/kernels/spectrogram_field.cu` | Spectrogram PTX | Create |
| `knowledge3d/cranium/kernels/hrtf_binaural.cu` | HRTF PTX | Create |

### 9.2 PTX Kernels Required

```cuda
// spectrogram_field.cu - Audio spectrogram to VectorDotMap
// hrtf_binaural.cu - Binaural audio rendering
// sdr_waterfall.cu - SDR signal visualization
// sonification.cu - Image to audio conversion
// temporal_spectrum.cu - Video temporal analysis
```

### 9.3 Integration Points

1. **Audio Galaxy** ↔ **VectorDotMap**: Share encoding
2. **Drawing Galaxy** ↔ **Sonification**: Visual → Audio
3. **Video Codec** ↔ **Audio Codec**: Synchronized encoding
4. **House/Galaxy** ↔ **Binaural**: Spatial positioning

---

## 10. Success Criteria

### Signal Processing
- [ ] STFT/ISTFT execute 100% on GPU (no numpy)
- [ ] Spectrogram → VectorDotMap achieves same compression as images
- [ ] SDR signals process at real-time rates (>1M samples/sec)

### Spatial Audio
- [ ] HRTF renders binaural audio < 5ms latency
- [ ] Room acoustics simulation in real-time
- [ ] Correct 3D positioning in Galaxy/House space

### Cross-Modal
- [ ] Audio spectrograms stored as VectorDotMap
- [ ] Sonification produces recognizable audio from images
- [ ] Cross-modal similarity search functional

### Integration
- [ ] Video codec uses unified signal pipeline
- [ ] Audio-video sync maintained in procedural encoding
- [ ] All signals shareable across Galaxies

---

## 11. References

### Core Implementation
- `knowledge3d/cranium/codecs/sovereign_ternary_audio_codec.py`
- `knowledge3d/cranium/codecs/sovereign_ternary_video_codec.py`
- `STRATEGY_AUDIO_AS_IMAGE_MULTIMODAL.md`

### Architecture Documents
- `docs/vocabulary/PROCEDURAL_VISUAL_SPECIFICATION.md` — VectorDotMap
- `docs/vocabulary/THREE_BRAIN_SYSTEM_SPECIFICATION.md` — Galaxy/House
- `docs/vocabulary/SPATIAL_UI_ARCHITECTURE_SPECIFICATION.md` — 3D space

### External Research
- Software Defined Radio (GNU Radio, rtl-sdr)
- HRTF databases (MIT KEMAR, CIPIC)
- Psychoacoustics (mel scale, loudness perception)

---

## 12. Conclusion

The Unified Signal Architecture treats all time-varying data as frequency components, enabling:

- **One codec for everything**: Audio, images, video, radio share VectorDotMap
- **Cross-modal discovery**: Find connections between sounds and images
- **Spatial integration**: 3D audio naturally fits Galaxy/House architecture
- **Infinite flexibility**: Same representation works from 20 Hz to 5 GHz

This is the signal foundation for K3D's multi-modal intelligence — where hearing, seeing, and sensing are unified operations on frequency-time data.

---

**Version History**:
- 1.0 (December 2025): Initial specification documenting frequency-time architecture, audio-image bridge, SDR integration, binaural audio, and cross-modal discovery.
