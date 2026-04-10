# K3D Procedural Texture Specification
## Adapting kkrieger's Werkkzeug1 for Sovereign GPU Engine

## 1. Introduction

### 1.1 Background: kkrieger's 96KB Breakthrough
**kkrieger** (2004, farbrausch) demonstrated extreme compression by replacing static assets with procedural generation algorithms:
- **No stored textures**: All textures generated at runtime via operator graphs
- **Procedural geometry**: CSG operations instead of stored meshes
- **Synthesized audio**: V2 music synthesizer with per-note waveforms
- **Total size**: 96KB executable containing full FPS game

**Key Insight**: Store algorithms, not data. A texture becomes a small DAG (Directed Acyclic Graph) of composable operators rather than megabytes of pixel data.

### 1.2 K3D Sovereignty Requirements
K3D's sovereign engine requires:
- **No external assets** in hot path (inference)
- **GPU-native texture generation** (CUDA texture arrays)
- **Deterministic reproduction** across systems
- **Physics integration** from visual properties
- **Stack-based RPN execution** for compact representation

## 2. Texture Operator Graph Architecture

### 2.1 Werkkzeug1 Operator Analysis
Werkkzeug1's texture system used a node-based editor where each texture was defined as a DAG of operators:

```
Source Operators → Transform Operators → Output Texture
       ↓                 ↓
  Perlin Noise        FFT Blur
  Voronoi Cells       Warp/Distort
  Constant Color      Color Adjust
                      Normal Map
```

**Key Characteristics**:
- Each operator takes 0-2 input textures plus parameters
- Graph evaluated depth-first at load time
- Operators maintain intermediate results until final "bake"
- Resolution-independent (generated at target resolution)

### 2.2 Adapted K3D Architecture

```
Galaxy Texture Star
├── visual_rpn (L1): [0xC0-0xDF opcodes]
├── meaning (L2): "rusted_metal_base"
└── physics_rpn_addr (L3): → friction=0.3
```

**Memory Layout**:
```
Texture Handle Pool (GPU):
  0: CUDA Texture Array Slot 0
  1: CUDA Texture Array Slot 1
  ...
  n: Intermediate Results (virtual)
  
RPN Stack (GPU Shared Memory):
  [float params] → [texture handles] → [result handle]
```

## 3. Texture RPN Opcodes (0xC0-0xDF)

### 3.1 Opcode Encoding Scheme
```
0xC0: TEX_PERLIN_NOISE    // fBm noise
0xC1: TEX_VORONOI         // Cellular patterns  
0xC2: TEX_FFT_BLUR        // Frequency-domain blur
0xC3: TEX_WARP            // Distortion mapping
0xC4: TEX_BLEND           // Alpha compositing
0xC5: TEX_NORMAL_MAP      // Height→normal conversion
0xC6: TEX_COLOR_RAMP      // Gradient mapping
0xC7: TEX_BAKE            // Finalize to GPU texture
0xC8: TEX_CELLULAR_NOISE  // Worley noise
0xC9: TEX_CHECKER         // Procedural checkerboard
0xCA: TEX_TURBULENCE      // Fractal turbulence
0xCB: TEX_MARBLE          // Marble vein pattern
0xCC: TEX_WOOD            // Wood grain
0xCD: TEX_METAL           // Anisotropic scratches
0xCE: TEX_GRADIENT        // Linear/radial gradient
0xCF: TEX_TRANSFORM       // Scale/rotate
```

### 3.2 Stack Semantics
Each opcode follows this pattern:
```
Parameters → Texture Handles → Result Handle
```

**Example: TEX_BLEND (0xC4)**:
```
Stack Before: [tex_a][tex_b][alpha][BLEND_MODE]
Execute:      Pops 4 values, pushes blended texture handle
Stack After:  [result_handle]
```

## 4. Core Texture Algorithms (Pseudocode)

### 4.1 TEX_PERLIN_NOISE (0xC0)
**Stack Input**: octaves(float), frequency(float), amplitude(float), persistence(float)

```cpp
// GPU Kernel: tex_perlin_noise_kernel
__global__ void perlin_noise(
    float* output, int width, int height,
    int octaves, float freq, float amp, float persistence)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    float total = 0.0f;
    float amplitude = amp;
    float frequency = freq;
    
    for (int i = 0; i < octaves; i++) {
        // Perlin's improved noise (gradient noise)
        float nx = x * frequency / width;
        float ny = y * frequency / height;
        
        // Quintic interpolation curve
        float fx = nx - floor(nx);
        float fy = ny - floor(ny);
        float u = fx * fx * fx * (fx * (fx * 6 - 15) + 10);
        float v = fy * fy * fy * (fy * (fy * 6 - 15) + 10);
        
        // Gradient hashing
        int g00 = hash(floor(nx), floor(ny));
        int g01 = hash(floor(nx), floor(ny)+1);
        int g10 = hash(floor(nx)+1, floor(ny));
        int g11 = hash(floor(nx)+1, floor(ny)+1);
        
        float d00 = dot(gradients[g00 & 15], vec2(fx, fy));
        float d01 = dot(gradients[g01 & 15], vec2(fx, fy-1));
        float d10 = dot(gradients[g10 & 15], vec2(fx-1, fy));
        float d11 = dot(gradients[g11 & 15], vec2(fx-1, fy-1));
        
        float x1 = lerp(d00, d10, u);
        float x2 = lerp(d01, d11, u);
        float result = lerp(x1, x2, v);
        
        total += result * amplitude;
        amplitude *= persistence;
        frequency *= 2.0f;
    }
    
    output[y*width + x] = total * 0.5f + 0.5f; // Map to [0,1]
}
```

### 4.2 TEX_VORONOI (0xC1)
**Stack Input**: cell_count(float), jitter(float)

```cpp
// GPU Kernel: tex_voronoi_kernel
__global__ void voronoi_noise(
    float* output, int width, int height,
    float cell_count, float jitter)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    float px = x / (float)width;
    float py = y / (float)height;
    
    // Scale to cell space
    float cell_x = px * cell_count;
    float cell_y = py * cell_count;
    
    int base_x = floor(cell_x);
    int base_y = floor(cell_y);
    
    float min_dist = FLT_MAX;
    
    // Check 3x3 neighborhood of cells
    for (int ny = -1; ny <= 1; ny++) {
        for (int nx = -1; nx <= 1; nx++) {
            int cell_idx_x = base_x + nx;
            int cell_idx_y = base_y + ny;
            
            // Generate jittered seed point
            vec2 seed;
            seed.x = cell_idx_x + hash_random(cell_idx_x, cell_idx_y, 0) * jitter;
            seed.y = cell_idx_y + hash_random(cell_idx_x, cell_idx_y, 1) * jitter;
            
            float dist = distance(vec2(cell_x, cell_y), seed);
            min_dist = fminf(min_dist, dist);
        }
    }
    
    output[y*width + x] = min_dist;
}
```

### 4.3 TEX_FFT_BLUR (0xC2)
**Stack Input**: texture_handle, radius(float), sigma(float)

```cpp
// GPU Pipeline: FFT-based separable convolution
// Step 1: Forward FFT of texture
cufftHandle plan;
cufftPlan2d(&plan, height, width, CUFFT_R2C);
cufftExecR2C(plan, input_real, output_complex);

// Step 2: Generate Gaussian kernel in frequency domain
__global__ void generate_gaussian_kernel(
    cufftComplex* kernel_ft, int width, int height,
    float sigma)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    // Gaussian in frequency domain is also Gaussian
    float fx = (x < width/2) ? x : x - width;
    float fy = (y < height/2) ? y : y - height;
    
    float r2 = (fx*fx)/(width*width) + (fy*fy)/(height*height);
    float gauss = exp(-2.0f * M_PI * M_PI * sigma * sigma * r2);
    
    kernel_ft[y*width + x].x = gauss;
    kernel_ft[y*width + x].y = 0.0f;
}

// Step 3: Multiply frequency domains
__global__ void multiply_complex(
    cufftComplex* a, cufftComplex* b, cufftComplex* result,
    int width, int height)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= width*height) return;
    
    // Complex multiplication
    float ar = a[idx].x, ai = a[idx].y;
    float br = b[idx].x, bi = b[idx].y;
    
    result[idx].x = ar*br - ai*bi;
    result[idx].y = ar*bi + ai*br;
}

// Step 4: Inverse FFT
cufftPlan2d(&inv_plan, height, width, CUFFT_C2R);
cufftExecC2R(inv_plan, multiplied_complex, output_real);
```

### 4.4 TEX_WARP (0xC3)
**Stack Input**: base_texture, warp_texture, intensity(float)

```cpp
// GPU Kernel: tex_warp_kernel
__global__ void warp_texture(
    float* output, float* base, float* warp,
    int width, int height, float intensity)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    // Get displacement from warp texture (RG channels)
    float dx = warp[(y*width + x)*4 + 0] * 2.0f - 1.0f;
    float dy = warp[(y*width + x)*4 + 1] * 2.0f - 1.0f;
    
    // Scale by intensity
    dx *= intensity;
    dy *= intensity;
    
    // Sample base texture with bilinear interpolation
    float sx = x + dx * width;
    float sy = y + dy * height;