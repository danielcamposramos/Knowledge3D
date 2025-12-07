# Universal Accessibility Specification

**Version**: 2.0
**Status**: Production
**License**: CC-BY-4.0 (Documentation), Apache 2.0 (Implementation)
**Date**: December 2025

---

## Abstract

This specification defines a unified, multi-modal accessibility model for spatial AI systems. It standardizes how text, Braille, sign language, haptics, and audio description are represented and accessed within a common 3D knowledge space so that assistive capabilities emerge from architecture rather than add-on tooling.

Normative requirements align with W3C WAI/WCAG guidance while leveraging WebXR, ARIA, and glTF 2.0 for implementation. The model is validated in production within Knowledge3D (K3D).

---

## 1. Goals and Scope

- Provide a modality-equal representation for text, tactile (Braille), visual, audio, and gesture data in one spatial model.
- Ensure zero-effort accessibility: the same node powers all modalities without manual duplication.
- Align with WCAG 2.2 and evolving WCAG 3.0 guidance using open web standards.

Non-goals: Define language-specific sign dictionaries or Braille contractions beyond Unicode mapping; these are implementation choices.

---

## 2. Core Model

### 2.1 K3D Node Accessibility Facets

For any knowledge node N at position (x, y, z), the following optional facets MAY be attached:

- Visual facet (human-readable text, imagery)
- AI facet (compressed representation for search/compute)
- Braille facet (Unicode Braille Patterns U+2800–U+28FF)
- Audio facet (narration or descriptions with timestamps/spatialization)
- Gesture facet (sign language via spatial action buffers)
- Haptic facet (controller/tactile feedback parameters)

Nodes MUST declare which facets are present in metadata for deterministic client behavior.

### 2.2 Dual-Texture Extension (Braille Layer)

Implementations SHOULD support a tri-UV map convention for text-bearing meshes:

- UV0: Visual text surface (for sighted users)
- UV1: AI compressed texture (for machine processing)
- UV2: Braille texture (Unicode Braille rendered to a tactile map)

The Braille layer MUST map 1:1 to the underlying text content and MUST be discoverable via ARIA metadata.

### 2.3 Sign Language via Action Buffers

Sign language gestures MUST be representable as fixed-width action buffers compatible with WebXR Hands API semantics, including:

- Action type identifier (e.g., GESTURE_SIGN_LANGUAGE)
- Hand positions and orientations (per hand)
- Finger joint poses (per finger, per joint)
- Timestamp and language metadata (ASL, BSL, Libras, JSL, etc.)

Gesture nodes MUST provide a human-renderable animation binding for playback on avatars.

---

## 3. Web Standards Alignment

- WCAG 2.2 (and 3.0 drafts): perceivable, operable, understandable, robust
- ARIA: `aria-label`, `aria-braillelabel` and related attributes
- WebXR: Hands Input Profiles for gesture capture and playback
- glTF 2.0: K3D extensions for extra facets and textures

Implementations MUST expose accessibility metadata through the same public interfaces used by non-accessible clients (no special/private APIs).

---

## 4. Compliance and Testing

### 4.1 Conformance Classes

- A: Visual + AI facets with labelled metadata
- AA: Adds Braille OR audio description
- AAA: Adds Braille AND audio AND sign gesture playback + haptics

### 4.2 Reference Tests

- Braille round-trip: text → Unicode Braille → rendered texture
- Gesture fidelity: buffer → avatar animation → user comprehension
- Audio description: synchronization and spatialization checks

Test artifacts are distributed as `.k3d` scenes with expected outputs.

---

## 5. Security and Privacy

Accessibility data (e.g., sign metadata) MAY imply personal preferences. Implementations SHOULD minimize collection and MUST respect user consent and local regulations.

---

## 6. IANA/Registry Considerations

Reserve identifiers for accessibility facets and action types in the K3D registry.

---

## 7. W3C Insertion Mapping

This specification operationalizes the contribution in `TEMP/W3C_INSERTION_7_UNIVERSAL_ACCESSIBILITY.md` and references dual-texture details in `TEMP/W3C_INSERTION_6_DUAL_TEXTURE_AND_MATRYOSHKA.md`.

---

## 8. Braille Galaxy — Procedural Tactile Representation

### 8.1 Architecture

Braille text is stored as **procedural dot patterns** in the Braille Galaxy, following the same RPN principles as other K3D data:

```
┌─────────────────────────────────────────────────────────┐
│                    BRAILLE GALAXY                        │
│                                                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐   │
│  │ Grade 1      │  │ Grade 2      │  │ Multi-       │   │
│  │ (Uncontracted)│  │ (Contracted) │  │ Language     │   │
│  │              │  │              │  │              │   │
│  │ a → ⠁        │  │ the → ⠮      │  │ EN, PT, JP   │   │
│  │ b → ⠃        │  │ and → ⠯      │  │ Libras dots  │   │
│  │ ...          │  │ ...          │  │ ...          │   │
│  └──────────────┘  └──────────────┘  └──────────────┘   │
│                          │                               │
│                          ▼                               │
│         ┌────────────────────────────────┐              │
│         │  PROCEDURAL DOT PATTERNS (RPN) │              │
│         │  6-dot or 8-dot configurations │              │
│         │  Stored as bit patterns        │              │
│         └────────────────────────────────┘              │
└─────────────────────────────────────────────────────────┘
```

### 8.2 Braille Cell as Procedural RPN

Each Braille cell is a **6-dot grid** (or 8-dot for computer Braille):

```
Standard 6-dot:        8-dot (Computer):
┌───┬───┐              ┌───┬───┐
│ 1 │ 4 │              │ 1 │ 4 │
├───┼───┤              ├───┼───┤
│ 2 │ 5 │              │ 2 │ 5 │
├───┼───┤              ├───┼───┤
│ 3 │ 6 │              │ 3 │ 6 │
└───┴───┘              ├───┼───┤
                       │ 7 │ 8 │
                       └───┴───┘
```

RPN encoding:
```rpn
# Braille cell encoding (6-dot as bit pattern)
# Letter 'a' = dot 1 only = 0b000001 = 0x01
BRAILLE_CELL 0x01                # Unicode ⠁ (U+2801)

# Letter 'b' = dots 1,2 = 0b000011 = 0x03
BRAILLE_CELL 0x03                # Unicode ⠃ (U+2803)

# Letter 'z' = dots 1,3,5,6 = 0b110101 = 0x35
BRAILLE_CELL 0x35                # Unicode ⠵ (U+2835)

# Contracted Braille: 'the' = single cell ⠮
BRAILLE_CONTRACTED "the" 0x2E    # U+282E
```

### 8.3 RPN Opcodes for Braille

```rpn
# Cell operations
BRAILLE_CELL pattern             # Create 6-dot cell from bit pattern
BRAILLE_CELL_8 pattern           # Create 8-dot cell from bit pattern
BRAILLE_UNICODE codepoint        # Create from Unicode (U+2800-U+28FF)

# Text conversion
TEXT_TO_BRAILLE grade language   # Convert text to Braille cells
  # grade: 1 (uncontracted), 2 (contracted)
  # language: en, pt, fr, de, jp, etc.
BRAILLE_TO_TEXT language         # Convert Braille back to text

# Spatial layout
BRAILLE_LINE cells[]             # Create horizontal line of cells
BRAILLE_BLOCK width height       # Create 2D Braille block
BRAILLE_POSITION x y             # Position in Galaxy space

# Rendering
BRAILLE_RENDER_TACTILE           # Generate haptic feedback pattern
BRAILLE_RENDER_VISUAL            # Render dots visually
BRAILLE_RENDER_AUDIO             # Sonify dot pattern (accessibility)

# Storage
BRAILLE_STORE id                 # Store in Braille Galaxy
BRAILLE_LOAD id                  # Load from Galaxy
```

### 8.4 Haptic Integration

Braille cells translate to haptic feedback for refreshable displays:

```rpn
# Haptic rendering for Braille displays
BRAILLE_LINE_CREATE text grade lang
HAPTIC_DEVICE_SELECT display_id
HAPTIC_BRAILLE_PUSH              # Push to refreshable display

# Vibration pattern for mobile devices
BRAILLE_TO_VIBRATION duration    # Convert cell pattern to vibration
  # Dot 1: 100ms buzz
  # Dot 2: 150ms buzz
  # Pause between dots: 50ms
  # Pause between cells: 200ms
VIBRATION_PLAY
```

### 8.5 Multi-Language Support

```rpn
# Language-specific Braille tables
BRAILLE_TABLE_LOAD "english-ueb"   # Unified English Braille
BRAILLE_TABLE_LOAD "portuguese-grade2"
BRAILLE_TABLE_LOAD "japanese-kana"
BRAILLE_TABLE_LOAD "libras-grade2"  # Brazilian Sign Language Braille

# Contract/expand
BRAILLE_CONTRACT "and" → ⠯       # English Grade 2
BRAILLE_CONTRACT "para" → ⠏      # Portuguese contraction
```

---

## 9. Sign Language Galaxy — Procedural Gesture Representation

### 9.1 Architecture

Sign language gestures are stored as **RPN sequences** describing hand positions, movements, and facial expressions:

```
┌─────────────────────────────────────────────────────────┐
│                  SIGN LANGUAGE GALAXY                    │
│                                                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐   │
│  │ ASL (USA)    │  │ Libras (BR)  │  │ BSL (UK)     │   │
│  │              │  │              │  │              │   │
│  │ hello → 🤚   │  │ olá → 🤟    │  │ hello → 👋   │   │
│  │ thank → 🙏   │  │ obrigado    │  │ thanks       │   │
│  └──────────────┘  └──────────────┘  └──────────────┘   │
│                          │                               │
│                          ▼                               │
│         ┌────────────────────────────────────┐          │
│         │  PROCEDURAL GESTURE SEQUENCES (RPN)│          │
│         │  Hand shape + Position + Movement  │          │
│         │  + Facial expression metadata      │          │
│         └────────────────────────────────────┘          │
│                          │                               │
│                          ▼                               │
│         ┌────────────────────────────────────┐          │
│         │     ANIMATION KEYFRAMES            │          │
│         │  WebXR Hands API compatible        │          │
│         │  Avatar-renderable                 │          │
│         └────────────────────────────────────┘          │
└─────────────────────────────────────────────────────────┘
```

### 9.2 Gesture Decomposition

Each sign decomposes into procedural components:

```
Sign = Handshape + Location + Movement + Orientation + Non-Manual
       (what)      (where)    (how)     (facing)       (face/body)
```

```rpn
# ASL "HELLO" sign
SIGN_BEGIN "hello" "ASL"
  HAND_SHAPE "flat_b"            # Hand configuration
  HAND_LOCATION "forehead"       # Starting position
  HAND_ORIENTATION "palm_out"    # Palm facing direction
  MOVEMENT_ARC "forward" 0.3     # Move forward from forehead
  FACIAL_EXPRESSION "neutral"    # Non-manual component
SIGN_END

# Libras "OLÁ" (hello)
SIGN_BEGIN "olá" "Libras"
  HAND_SHAPE "five"
  HAND_LOCATION "chin"
  MOVEMENT_WAVE 2                # Wave motion, 2 cycles
  FACIAL_EXPRESSION "friendly"
SIGN_END
```

### 9.3 RPN Opcodes for Sign Language

```rpn
# Sign definition
SIGN_BEGIN word language         # Start sign definition
SIGN_END                         # End sign definition
SIGN_COMPOSITE signs[]           # Combine multiple signs

# Hand shape (ASL handshapes)
HAND_SHAPE shape                 # flat_b, fist, five, claw, hook, etc.
HAND_CONFIG fingers[]            # Custom finger configuration
  # Each finger: [extended, bent, closed]
FINGER_SPELL letter              # Finger spelling

# Location (relative to body)
HAND_LOCATION area               # forehead, chin, chest, side, neutral
HAND_POSITION x y z              # Precise 3D position
HAND_CONTACT surface             # Touch location on body/face

# Movement
MOVEMENT_LINEAR dx dy dz         # Straight line movement
MOVEMENT_ARC direction angle     # Curved path
MOVEMENT_CIRCLE radius direction # Circular motion
MOVEMENT_WAVE cycles             # Oscillating motion
MOVEMENT_ZIGZAG count            # Back and forth
MOVEMENT_REPEAT count            # Repeat last movement

# Orientation
HAND_ORIENTATION facing          # palm_in, palm_out, palm_up, palm_down
WRIST_ROTATION angle             # Twist wrist
HAND_TILT pitch roll             # Tilt hand

# Two-handed signs
SIGN_TWO_HANDED
DOMINANT_HAND                    # Define dominant hand actions
NON_DOMINANT_HAND                # Define other hand (often static)
HAND_RELATION relation           # mirrored, stacked, clasped, etc.

# Non-manual markers
FACIAL_EXPRESSION type           # neutral, question, negation, emphasis
EYEBROW_POSITION position        # raised, lowered, furrowed
LIP_PATTERN pattern              # Mouth shapes for specific signs
HEAD_MOVEMENT type               # nod, shake, tilt
BODY_LEAN direction              # forward, back, left, right

# Animation output
SIGN_TO_KEYFRAMES fps            # Generate animation keyframes
SIGN_TO_WEBXR                    # Export to WebXR Hands format
SIGN_TO_AVATAR avatar_id         # Render on avatar
```

### 9.4 Finger Spelling

```rpn
# ASL Manual Alphabet
FINGER_SPELL_BEGIN language
  # A = fist, thumb to side
  LETTER 'A' HAND_SHAPE "fist" THUMB_POSITION "side"
  # B = flat hand, thumb across palm
  LETTER 'B' HAND_SHAPE "flat" THUMB_POSITION "across"
  # ... all 26 letters defined
FINGER_SPELL_END

# Spell a word
WORD_TO_FINGERSPELL "hello" "ASL"
# Generates: H → E → L → L → O with transitions
```

### 9.5 Sign Language to Audio Description

Integration with unified signal architecture:

```rpn
# Generate audio description of sign
SIGN_LOAD "hello" "ASL"
SIGN_DESCRIBE                    # Generate text description
  # "Right hand flat, palm forward at forehead, moves away"
TEXT_TO_SPEECH                   # Convert to audio
BINAURAL_POSITION 0 1.5 -1       # Position in 3D space
```

### 9.6 Avatar Animation

```rpn
# Render sign on avatar
SIGN_SEQUENCE_BEGIN
  SIGN_LOAD "hello"
  PAUSE_MS 500
  SIGN_LOAD "my"
  SIGN_LOAD "name"
  FINGER_SPELL "Daniel"
SIGN_SEQUENCE_END

# Export to avatar
AVATAR_SKELETON_LOAD "humanoid"
SIGN_TO_SKELETON                 # Map sign keyframes to skeleton
ANIMATION_EXPORT "greeting.glb"  # Export as glTF animation
```

### 9.7 Supported Sign Languages

| Language | Code | Features |
|----------|------|----------|
| ASL | `ASL` | American Sign Language |
| Libras | `Libras` | Brazilian Sign Language |
| BSL | `BSL` | British Sign Language |
| LSF | `LSF` | French Sign Language |
| DGS | `DGS` | German Sign Language |
| JSL | `JSL` | Japanese Sign Language |
| Auslan | `Auslan` | Australian Sign Language |

---

## 10. Spatial Audio for Accessibility

### 10.1 Audio Description with Binaural Positioning

Audio descriptions are positioned in 3D space to match visual content:

```rpn
# Scene with spatial audio descriptions
SCENE_LOAD "museum_exhibit"

# Describe object at specific location
AUDIO_DESCRIPTION "A bronze statue of a horse"
  POSITION 2.0 1.5 -3.0          # 3D position in scene
  HRTF_RENDER                    # Binaural audio

# As user moves, audio updates
LISTENER_POSITION_UPDATE x y z
AUDIO_DESCRIPTIONS_UPDATE        # Re-render all with new perspective
```

### 10.2 Sonification of Visual UI

```rpn
# Convert visual interface to audio
UI_ELEMENT_FOCUS "submit_button"
  POSITION_TO_STEREO_PAN         # Button position → stereo pan
  ELEMENT_TYPE_TO_PITCH          # Button type → pitch
  STATE_TO_TIMBRE                # Enabled/disabled → sound quality
SONIFICATION_PLAY

# Navigation sounds
UI_NAVIGATION_MODE
  HOVER_SOUND pitch              # Pitch varies with vertical position
  SELECT_SOUND                   # Confirmation sound
  ERROR_SOUND                    # Rejection sound
```

### 10.3 Braille + Audio Multimodal

```rpn
# Combined Braille and audio output
TEXT_LOAD "Welcome to K3D"
TEXT_TO_BRAILLE 2 "en"           # Grade 2 English Braille
BRAILLE_TO_HAPTIC                # Send to Braille display
TEXT_TO_SPEECH                   # Also speak it
AUDIO_SPATIAL_SYNC               # Sync audio timing with Braille refresh
```

---

## 11. Implementation

### 11.1 Files to Create

| File | Purpose |
|------|---------|
| `knowledge3d/cranium/braille_galaxy.py` | Braille procedural storage |
| `knowledge3d/cranium/sign_language_galaxy.py` | Sign gesture storage |
| `knowledge3d/cranium/accessibility_audio.py` | Spatial audio descriptions |
| `knowledge3d/cranium/kernels/braille_render.cu` | GPU Braille rendering |
| `knowledge3d/cranium/kernels/sign_keyframes.cu` | Gesture to animation |

### 11.2 PTX Kernels

```cuda
// braille_render.cu - Braille cell to visual/haptic
// sign_keyframes.cu - Gesture RPN to animation keyframes
// spatial_description.cu - Audio description HRTF rendering
// sonification_ui.cu - UI element sonification
```

### 11.3 Success Criteria

- [ ] Braille cells encode/decode via RPN (100% on GPU)
- [ ] Sign language gestures render to avatar in real-time
- [ ] Grade 1 and Grade 2 Braille for EN, PT, Libras
- [ ] ASL, Libras, BSL gesture libraries
- [ ] Spatial audio descriptions with HRTF
- [ ] Braille display haptic output functional

---

## 12. Attribution & Academic Context

**For complete attributions**, see [ATTRIBUTIONS.md](../../ATTRIBUTIONS.md) in the K3D repository.

**Key Credits**:

1. **W3C WAI (Web Accessibility Initiative)**:
   - WCAG 2.2 guidelines (ISO/IEC 40500:2025)
   - ARIA standards for semantic markup
   - Foundation for K3D's spatial accessibility

2. **Multi-Modal Fusion Research**:
   - Cross-modal alignment (text, image, audio, haptics)
   - K3D implements organic spatial co-location
   - 98.05% RLWHF accuracy on multi-modal tasks

3. **Unicode Braille Standards**:
   - Braille pattern encoding (U+2800-U+28FF)
   - K3D implements GPU-native Braille rendering

4. **Game Industry** (Accessibility):
   - Audio-only game design principles
   - Haptic feedback systems
   - Spatial audio techniques

5. **WebXR Accessibility**:
   - VR/AR accessibility guidelines
   - K3D extends with zero-config multi-modal support

K3D's Universal Accessibility specification is a novel contribution that makes accessibility a first-class architectural property through spatial multi-modal integration.

---

## 13. Related Specifications

- **[UNIFIED_SIGNAL_SPECIFICATION.md](UNIFIED_SIGNAL_SPECIFICATION.md)** — Audio, SDR, binaural processing
- **[PROCEDURAL_VISUAL_SPECIFICATION.md](PROCEDURAL_VISUAL_SPECIFICATION.md)** — Drawing Galaxy, VectorDotMap
- **[THREE_BRAIN_SYSTEM_SPECIFICATION.md](THREE_BRAIN_SYSTEM_SPECIFICATION.md)** — Galaxy/House architecture
- **[DUAL_CLIENT_CONTRACT_SPECIFICATION.md](DUAL_CLIENT_CONTRACT_SPECIFICATION.md)** — Human + AI dual representation

---

## 14. Version History

- **2.0** (December 2025): Added Braille Galaxy (Section 8), Sign Language Galaxy (Section 9), Spatial Audio Accessibility (Section 10), Implementation details (Section 11)
- **1.0** (November 2025): Initial specification with facet model and compliance classes

