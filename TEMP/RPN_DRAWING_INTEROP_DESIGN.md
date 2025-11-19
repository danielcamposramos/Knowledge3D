# RPN ⇄ Drawing Kernel Interop Design

## Goal
Leverage the existing 18-stack, 69-depth RPN math engine to feed procedural drawing without hardcoding advanced math (sin/cos/atan2) in the drawing PTX. Math stays in the RPN kernel; drawing stays lean, consuming precomputed values via shared GPU buffers.

## Handoff Protocol
1. Host prepares/compiles two programs:
   - `rpn_program`: computes required values (e.g., cos/sin pairs, arc sample points, rotation matrices).
   - `draw_program`: consumes precomputed buffers via dedicated opcodes.
2. RPN kernel executes first, writing outputs to a GPU buffer handle (device pointer).
3. Drawing kernel launches with:
   - Pointer to bytecode
   - Pointer(s) to precomputed buffer(s)
   - Optional metadata (counts/strides)

## Buffer Formats
- **Rotation matrix buffer**: contiguous float32 pairs `(cos, sin)`; length = N rotations.
- **Arc/ellipse point buffer**: float32 XY pairs; shape `(num_points, 2)`. Stored as `[x0,y0, x1,y1, ...]`.
- **Generic value buffer**: float32 array; consumer opcode interprets layout.

## New Opcodes (design)
### ROTATE_MATRIX (0x7A)
Inputs: cos, sin (precomputed). Applies rotation to current transform (or multiplies into matrix).
```
OPC_ROTATE_MATRIX:
  load cos, sin from bytecode or buffer
  // compose into current 2x3 affine: [a c e; b d f]
  // new_a = a*cos + c*sin
  // new_c = -a*sin + c*cos
  // new_b = b*cos + d*sin
  // new_d = -b*sin + d*cos
  // e,f unchanged
```

### PRECOMPUTED_PATH (0x7B)
Inputs: buffer_id/index, count (optional stride). Emits segments by walking the precomputed XY pairs.
```
OPC_PRECOMPUTED_PATH:
  addr = base_ptr_for_buffer(buffer_id)
  for i in 0..count-1:
    x = addr[2*i+0]; y = addr[2*i+1];
    // apply current transform, emit segment from (prev→current)
```

## Execution Flow Example
1) RPN: compute 32-point arc
```
   rpn_program = "0 2PI RANGE 32 STEPS DUP COS SWAP SIN"
   -> writes 32 xy pairs into buffer ArcBuf0
```
2) Drawing: consume precomputed path
```
   draw_program = "PRECOMPUTED_PATH ArcBuf0 STROKE"
```

## Notes
- Keep buffers opaque to host; only device pointers are passed.
- Transform matrix in drawing kernel still applied to consumed points.
- Local ternary hints remain supported; they can modulate tessellation counts even for precomputed paths (e.g., pick fewer/more points from buffer).
