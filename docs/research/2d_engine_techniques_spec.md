# K3D Sovereign GPU Engine: 2D-to-3D Extension Specification

**Document Status:** Research Spec | **Version:** 0.1.0  
**Path:** `/K3D/GitHub/Knowledge3D/docs/research/2d_engine_techniques_spec.md`

## Executive Summary

This specification details how classical 2D game engine techniques map to K3D's sovereign GPU architecture, where 2D primitives are special cases of 3D operations via RPN transforms. The K3D engine treats 2D as `z=0` projection of 3D space, enabling seamless dimensional extension.

## 1. Rendering Techniques Mapping

### 1.1 Signed Distance Fields (SDFs)

**2D Technique:** Using SDFs for crisp sprites at any scale with minimal texture memory.
- **PICO-8 Insight:** Their "stretchy sprites" use software scaling, but SDFs provide GPU-native resolution independence
- **GPU Adaptation:** Store SDFs in 8-bit textures, evaluate at fragment shader

**K3D Implementation:**
```rpn
# Drawing Galaxy SDF opcodes (0x80-0x9F extended range)
0x80: SDF_CIRCLE        # d = length(p) - r
0x81: SDF_BOX           # d = max(abs(p)-size,0)
0x82: SDF_ROUNDED_BOX   # d = length(max(abs(p)-size,0)) - r
0x83: SDF_UNION         # d = min(d1, d2)
0x84: SDF_INTERSECTION  # d = max(d1, d2)
0x85: SDF_SUBTRACTION   # d = max(d1, -d2)
0x86: SDF_SMOOTH_UNION  # d = lerp(d1,d2,k) - k*min(0,|d1-d2|)
0x87: SDF_PALETTE_LOOKUP # Map SDF distance to palette index
```

**2D→3D Extension:** A 2D SDF circle becomes a 3D sphere via `SDF_CIRCLE` + `EXTRUDE_Z` transform:
```rpn
# 2D circle at z=0
PUSH 0.5           # radius
SDF_CIRCLE         # Stack: d_circle
DRAW_SDF           # Renders 2D disc

# 3D sphere via extrusion
PUSH 0.5           # radius
SDF_CIRCLE         # d_circle(x,y)
EXTRUDE_Z 0.5      # Stack: d_sphere = sqrt(d_circle² + z²) - 0.5
DRAW_SDF_3D        # Renders 3D sphere
```

### 1.2 Palette Swapping

**2D Technique:** 16-color palette indexing for rapid sprite recoloring (PICO-8/TIC-80).
- **Classic Approach:** 4-bit color indices, runtime palette swap tables

**K3D Implementation:**
```rpn
# Palette Star in Drawing Galaxy
0x70: PALETTE_CREATE    # Creates palette with 16 entries
0x71: PALETTE_SWAP      # Swaps palette indices
0x72: COLOR_REMAP       # Maps color ranges via RPN program

# Example: PICO-8 16-color palette mapped to RGBA8888
PALETTE_CREATE 16
PUSH 0x000000  PALETTE_SET 0   # Black
PUSH 0x1D2B53  PALETTE_SET 1   # Dark blue
PUSH 0x7E2553  PALETTE_SET 2   # Dark purple
# ... 13 more colors
```

**2D→3D Extension:** Palette stars become material properties in 3D:
```rpn
# 2D palette index becomes 3D material property
FETCH_SPRITE 0x42      # Get sprite with palette indices
APPLY_PALETTE 1        # Apply alternative palette
EXTRUDE_Z 0.1          # Becomes 3D relief with material
SET_MATERIAL_ATTR diffuse_color PALETTE_FETCH
```

### 1.3 Sprite Batching & Tilemap Rendering

**2D Technique:** GPU instancing for tilemaps, quad batching for sprites.
- **Raylib Insight:** `DrawTexturePro()` with source/dest rectangles
- **Optimization:** Sort by texture, z-depth, blend mode

**K3D Implementation:**
```rpn
# Tilemap opcodes (0x90-0x9F)
0x90: TILEMAP_CREATE    # width, height, tile_size
0x91: TILEMAP_SET       # x, y, tile_id
0x92: TILEMAP_BATCH     # Render entire map with instancing
0x93: TILEMAP_FETCH     # Get tile at position → RPN program
0x94: TILEMAP_AUTO_TILE # Applies autotiling rules

# Example: 16x16 tilemap with autotiling
TILEMAP_CREATE 32 32 16  # 32x32 tiles, 16px each
LOAD_TILESET "dungeon.tiles"
TILEMAP_AUTO_TILE 0 0 32 32  # Apply bitmask autotiling
TILEMAP_BATCH              # GPU instance render
```

**2D→3D Extension:** Tilemaps become voxel slices or wall extrusions:
```rpn
# 2D tile becomes 3D wall section
TILEMAP_FETCH 5 10      # Get tile at (5,10)
SWAP                    # Tile data on stack
EXTRUDE_Z 2.0           # Extrude to 3D wall of height 2.0
APPLY_TEXTURE WRAP_X WRAP_Y
```

### 1.4 Scanline & CRT Effects

**2D Technique:** Post-processing effects for retro aesthetics.
- **PICO-8 Emulation:** Scanline overlay, color bleed, phosphor decay

**K3D Implementation:**
```rpn
# Post-processing opcodes (0xA0-0xAF)
0xA0: POSTPROCESS_SCANLINE    # interval, brightness
0xA1: POSTPROCESS_CRT_CURVE   # curvature, vignette
0xA2: POSTPROCESS_CHROMA      # RGB separation
0xA3: POSTPROCESS_PIXELATE    # pixel size

# Example: CRT effect chain
PUSH 3 POSTPROCESS_SCANLINE    # Every 3rd line darker
PUSH 0.1 POSTPROCESS_CHROMA    # Slight color separation
PUSH 0.3 POSTPROCESS_CRT_CURVE # Screen curvature
```

### 1.5 Parallax Backgrounds

**2D Technique:** Multiple layers moving at different speeds.
- **Classic Approach:** Camera-relative layer transforms

**K3D Implementation:**
```rpn
# Parallax opcodes (0xB0-0xBF)
0xB0: PARALLAX_LAYER_CREATE  # texture, speed_factor
0xB1: PARALLAX_UPDATE        # camera_position
0xB2: PARALLAX_BATCH         # Render all layers

# Example: 3-layer parallax
PARALLAX_LAYER_CREATE "sky.png" 0.1
PARALLAX_LAYER_CREATE "mountains.png" 0.5  
PARALLAX_LAYER_CREATE "foreground.png" 1.0
PUSH camera_x PUSH camera_y PARALLAX_UPDATE
```

**2D→3D Extension:** Parallax becomes true 3D layering:
```rpn
# 2D parallax becomes 3D particle volume
PARALLAX_LAYER_CREATE "stars.png" 0.05
EXTRUDE_Z 100.0                 # Stars at varying depths
APPLY_DEPTH_TEST LESS          # True 3D sorting
```

## 2. Physics System Extension

### 2.1 Physics Opcode Range Allocation

**Proposed Mapping:**
```
0x130-0x14F: 2D Physics Sub-range
0x150-0x17F: 3D Physics (existing)

0x130: PHYS2D_AABB_CREATE      # x,y,w,h
0x131: PHYS2D_AABB_SWEEP        # dx,dy → time, normal
0x132: PHYS2D_SLOPE_CORRECTION  # angle, step_height
0x133: PHYS2D_COYOTE_TIME       # grace_period
0x134: PHYS2D_JUMP_BUFFER       # buffer_window
0x135: PHYS2D_ONE_WAY_PLATFORM  # direction, pass_through
0x136: PHYS2D_PUSH_SLIDE        # velocity, mass_ratio
0x137: PHYS2D_VERLET_INTEGRATE  # points, constraints
0x138: PHYS2D_COLLISION_GROUP   # bitmask
0x139: PHYS2D_RAYCAST_2D        # origin, direction → hit
```

### 2.2 Platformer Physics Implementation

**AABB Sweep with Slope Correction:**
```rpn
# Platformer character movement
PHYS2D_AABB_CREATE player_x player_y 1.0 2.0  # 1x2 AABB
PUSH velocity_x PUSH velocity_y
PHYS2D_AABB_SWEEP                      # Sweep test
DUP                                    # Copy collision time
PUSH 0.0 GREATER                       # if collision_time > 0
JUMP_IF_NO_COLLISION                   # Skip correction

# Slope handling
PHYS2D_SLOPE_CORRECTION 45.0 0.25     # 45° max, 0.25 step
APPLY_GRAVITY 9.8
PHYS2D_COYOTE_TIME 0.15               # 150ms coyote time
PHYS2D_JUMP_BUFFER 0.2                # 200ms jump buffer
```

**2D→3D Extension:** AABB becomes 3D AABB with height:
```rpn
# 2D platformer AABB → 3D collision capsule
PHYS2D_AABB_CREATE x y w h
EXTRUDE_Z height                     # Extrude to 3D
PHYS3D_CAPSULE_CREATE radius height # Convert to 3D capsule
```

### 2.3 Top-Down Physics (Push/Slide)

```rpn
# Top-down collision response
PHYS2D_AABB_CREATE entity_x entity_y 1.0 1.0
PUSH push_x PUSH push_y
PHYS2D_PUSH_SLIDE                    # Push away from obstacles
PHYS2D_COLLISION_GROUP 0x01         # Player group
PHYS2D_COLLISION_MASK 0x02          # Collide with obstacle group
```

### 2.4 Verlet Integration for Cloth/Ropes

```rpn
# 2D verlet cloth (10x10 grid)
PHYS2D_VERLET_CREATE 100            # 100 points
FOR i 0 100                         # Initialize grid
  PUSH i%10 PUSH i/10               # x,y position
  PHYS2D_VERLET_SET_POSITION i
NEXT

# Add constraints (structural, shear, bend)
FOR i 0 90                          # Structural horizontal
  PHYS2D_VERLET_ADD_CONSTRAINT i i+1 1.0
NEXT
# ... more constraints

# Simulation step
PHYS2D_VERLET_INTEGRATE 0.016       # 60 FPS timestep
PHYS2D_VERLET_SOLVE_CONSTRAINTS 3   # 3 iterations
```

**2D→3D Extension:** 2D verlet cloth becomes 3D softbody:
```rpn
PHYS2D_VERLET_CREATE 100
EXTRUDE_Z 1.0                        # Add z-coordinate
PHYS3D_SOFTBODY_CONVERT              # Convert to 3D softbody
PHYS3D_SOFTBODY_ADD_VOLUME_CONSTRAINT
```

## 3. Minimal Engine Analysis & Adaptation

### 3.1 PICO-8 (128×128, 16 colors, Lua)

**Key Algorithms to Port:**
1. **Fixed-point math:** 16.16 fixed point for position/velocity
2. **Camera culling:** Simple AABB in viewport check
3. **Sprite flipping:** X/Y mirror via negative scale
4. **Map storage:** 128×32 cells, 2 bytes per cell (tile+flags)

**K3D Adaptation:**
```rpn
# PICO-8 fixed-point emulation
0xC0: FIXED_CONVERT      # float → fixed(16.16)
0xC1: FIXED_ADD          # a + b
0xC2: FIXED_MUL          # a * b
0xC3: FIXED_DIV          # a / b

# PICO-8 camera (0,0 to 128,128)
CAMERA_SET 0 0 128 128
SPRITE_DRAW sprite_id x y [flip_x] [flip_y] [scale]
MAP x y w h sx sy [layer]  # Draw tilemap region
```

### 3.2 TIC-80

**Key Features:**
- **Sprite sheet:** 240×136, 8×8 sprites
- **Bank switching:** 8 memory banks for assets
- **Scanline callback:** Per-line raster effects

**K3D Adaptation:**
```rpn
0xD0: SPRITESHEET_LOAD   # Load TIC-80 format spritesheet
0xD1: BANK_SWITCH        # Switch asset bank
0xD2: SCANLINE_CALLBACK  # Register per-line callback
```

### 3.3 Raylib 2D & Love2D

**Key Insights:**
- **Batch rendering:** Love2D's SpriteBatch for dynamic sprites
- **Shader effects:** Raylib's post-processing shader system
- **Immediate mode:** Simple draw calls for prototyping

**K3D Implementation:**
```rpn
# Love2D-style batch
0xE0: SPRITE_BATCH_CREATE
0xE1: SPRITE_BATCH_ADD    # sprite, x, y, rotation, scale
0xE2: SPRITE_BATCH_DRAW

# Raylib-style immediate mode
DRAW_TEXTURE texture_id x y width height [color]
DRAW_TEXTURE_PRO texture_id source_rect dest_rect origin rotation [color]
```

## 4. Procedural Generation via Grammar Galaxy

### 4.1 Wave Function Collapse as RPN Constraint Propagation

**WFC Algorithm Mapping:**
```
Grammar Galaxy Equivalent:
L1: Tile patterns (visual form)
L2: Adjacency rules (semantic meaning)
L3: Constraint propagation (RPN rules)
L4: Backtracking strategy (meta-rules)
```

**K3D Implementation:**
```rpn
# WFC opcodes in Grammar Galaxy (0x300-0x31F)
0x300: WFC_INIT_GRID        # width, height, tile_set
0x301: WFC_ADD_CONSTRAINT   # tile_a, direction, tile_b
0x302: WFC_PROPAGATE        # Propagate constraints
0x303: WFC_COLLAPSE         # Collapse lowest entropy cell
0x304: WFC_BACKTRACK        # Backtrack on contradiction
0x305: WFC_GENERATE         # Generate full output

# Example: Dungeon generation
WFC_INIT_GRID 32 32 dungeon_tiles
WFC_ADD_CONSTRAINT FLOOR RIGHT WALL   # Floor can't have wall to right
WFC_ADD_CONSTRAINT WALL LEFT DOOR    # Wall can have door to left
# ... more adjacency rules
WFC_GENERATE                          # Run WFC
```

**2D→3D Extension:** 2D tile constraints become 3D voxel constraints:
```rpn
WFC_INIT_GRID_3D 16 16 16 voxel_tiles  # 3D voxel grid
WFC_ADD_CONSTRAINT_3D WALL ABOVE AIR    # Wall must have air above
WFC_ADD_CONSTRAINT_3D FLOOR BELOW WALL # Floor must have wall below
```

### 4.2 BSP Dungeon Generation

```rpn
# BSP tree generation as RPN program
0x310: BSP_SPLIT          # node, axis, position
0x311: BSP_LEAF_ROOM      # leaf, min_size
0x312: BSP_CONNECT_ROOMS  # Connect leaves with corridors

# Example: Generate dungeon
BSP_CREATE 0 0 64 64      # Root node
BSP_SPLIT HORIZONTAL 32   # Split horizontally
BSP_SPLIT VERTICAL 16     # Split left child vertically
BSP_SPLIT VERTICAL 48     # Split right child vertically
BSP_LEAF_ROOMS 6 6        # Create rooms in leaves (min 6x6)
BSP_CONNECT_ROOMS         # Connect with corridors
```

### 4.3 L-Systems for 2D Plants

```rpn
# L-system opcodes
0x320: LSYSTEM_INIT       # axiom, rules
0x321: LSYSTEM_ITERATE    # iterations
0x322: LSYSTEM_DRAW_2D    # Draw as 2D plant
0x323: LSYSTEM_EXTRUDE_3D # Extrude to 3D

# Example: Fractal tree
LSYSTEM_INIT "F"          # Axiom: F
LSYSTEM_ADD_RULE "F" "F[+F][-F]"  # Rewrite rule
LSYSTEM_ITERATE 4         # 4 iterations
LSYSTEM_DRAW_2D           # Draw 2D tree
LSYSTEM_EXTRUDE_3D 0.1    # Extrude to 3D bark texture
```

## 5. Key Insight: 2D→3D Extension Patterns

### 5.1 Dimensional Extension Operators

**Core RPN Opcodes for Dimensional Extension:**
```
0xF0: EXTRUDE_Z           # height → makes 2D shape 3D
0xF1: REVOLVE_Y           # angle → rotates around Y axis
0xF2: SWEEP_PATH          # path_rpn → sweeps along path
0xF3: LOFT_PROFILES       # profiles[] → skin between profiles
```

### 5.2 2D Primitive → 3D Solid Mappings

| 2D Primitive | 3D Extension | RPN Transformation |
|--------------|--------------|-------------------|
| Circle       | Sphere       | `SDF_CIRCLE → EXTRUDE_Z → SDF_SPHERE` |
| Rectangle    | Box          | `RECT → EXTRUDE_Z → BOX` |
| Line         | Cylinder     | `LINE → SWEEP_CIRCLE → CYLINDER` |
| Polygon      | Prism        | `POLYGON → EXTRUDE_Z → PRISM` |
| Tile         | Voxel        | `TILE → EXTRUDE_Z → VOXEL` |
| Sprite       | Billboard    | `SPRITE → BILLBOARD_3D` |

### 5.3 Camera & Projection Modes

```rpn
# Camera mode switching
CAMERA_MODE_2D            # Orthographic, no perspective
CAMERA_MODE_3D            # Perspective projection
CAMERA_MODE_ISOMETRIC     # Isometric (2.5D)

# Example: 2.5D isometric game
CAMERA_MODE_ISOMETRIC
SET_VIEW_ANGLE 30 45      # 30° elevation, 45° rotation
TILEMAP_CREATE_3D 32 32 8 # 3D tilemap for height
```

## 6. Performance Considerations

### 6.1 Hot Path Optimization

**PTX Kernel Specialization:**
```ptx
// SDF evaluation kernel (hot path)
.func SDF_EVAL(.param .b64 sdf_program) {
  // Stack-based RPN evaluator in registers
  // Early exit for empty fragments
}

// Tilemap instancing kernel
.func TILEMAP_BATCH(.param .b64 tile_data, .param .b64 instance_data) {
  // One thread per tile, shared memory for texture cache
}
```

### 6.2 Memory Hierarchy for 2D Assets

```
L0: Registers      - Active sprite data, SDF stack
L1: Shared Memory  - Tile cache, palette tables  
L2: L2 Cache       - Tilemaps, sprite sheets
VRAM:              - Textures, vertex buffers
System RAM:        - Asset packages, WFC constraints
```

### 6.3 Asynchronous Pipeline

```
Frame N:
1. Physics 2D (0x130-0x14F) parallel with 3D physics
2. WFC generation (Grammar Galaxy, async)
3. Sprite batching (Drawing Galaxy, parallel instances)
4. Post-processing (scanlines, CRT, async compute)
```

## 7. Reference Implementation Roadmap

### Phase 1: Core 2D System (v0.8.0)
- [ ] SDF primitive opcodes (0x80-0x9F)
- [ ] Physics 2D sub-range (0x130-0x14F)
- [ ] Tilemap batching (0x90-0x9F)
- [ ] PICO-8 palette compatibility

### Phase 2: Procedural Generation (v0.9.0)
- [ ] WFC in Grammar Galaxy (0x300-0x31F)
- [ ] BSP dungeon generation
- [ ] L-systems for vegetation

### Phase 3: 2D→3D Bridge (v1.0.0)
- [ ] Extrusion operators (0xF0-0xF3)
- [ ] Isometric camera mode
- [ ] 2D sprite → 3D billboard conversion

## 8. Testing & Validation

### Test Suite:
1. **Celeste Classic Physics Test:** Implement Celeste's player movement
2. **PICO-8 Cartridge Loader:** Load and run PICO-8 .p8 files
3. **WFC Consistency Test:** Verify procedural generation constraints
4. **Performance Benchmark:** 10k sprites at 60 FPS target

## Conclusion

K3D's sovereign GPU engine uniquely positions 2D techniques as special cases of 3D operations, enabled by RPN transform stacks. This specification provides a roadmap for implementing classical 2D game engine techniques while maintaining the architectural purity of K3D's galaxy-based design. The key innovation is treating dimensionality as an operator (`EXTRUDE_Z`, `REVOLVE_Y`) rather than separate rendering pipelines, enabling seamless transitions between 2D and 3D representations.

**Next Steps:** Implement prototype SDF renderer and physics 2D sub-range, then validate with PICO-8 compatibility tests.