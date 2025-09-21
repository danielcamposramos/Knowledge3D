extern "C" __global__ void generate_shape_kernel(
    const float* __restrict__ embedding,
    float* __restrict__ vertices,
    unsigned int vertex_count,
    unsigned int shape_type
) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= vertex_count) {
        return;
    }

    float scale = fabsf(embedding[0] + embedding[1] + embedding[2]);
    if (scale < 1e-3f) {
        scale = 1.0f;
    }

    float vx = 0.0f;
    float vy = 0.0f;
    float vz = 0.0f;

    switch (shape_type) {
        case 0: { // tetrahedron
            if (tid == 0) { vx = 1.0f;  vy = 1.0f;  vz = 1.0f; }
            else if (tid == 1) { vx = -1.0f; vy = -1.0f; vz = 1.0f; }
            else if (tid == 2) { vx = -1.0f; vy = 1.0f; vz = -1.0f; }
            else if (tid == 3) { vx = 1.0f;  vy = -1.0f; vz = -1.0f; }
            break;
        }
        case 1: { // cube
            int sx = (tid & 1) ? 1 : -1;
            int sy = (tid & 2) ? 1 : -1;
            int sz = (tid & 4) ? 1 : -1;
            vx = (float)sx;
            vy = (float)sy;
            vz = (float)sz;
            break;
        }
        case 2: { // octahedron
            if (tid == 0) { vx = 1.0f; }
            else if (tid == 1) { vx = -1.0f; }
            else if (tid == 2) { vy = 1.0f; }
            else if (tid == 3) { vy = -1.0f; }
            else if (tid == 4) { vz = 1.0f; }
            else if (tid == 5) { vz = -1.0f; }
            break;
        }
        case 3: { // icosahedron
            const float phi = 1.6180339887498948482f;
            switch (tid) {
                case 0:  vx = phi;  vy = 1.0f;  vz = 0.0f; break;
                case 1:  vx = -phi; vy = 1.0f;  vz = 0.0f; break;
                case 2:  vx = phi;  vy = -1.0f; vz = 0.0f; break;
                case 3:  vx = -phi; vy = -1.0f; vz = 0.0f; break;
                case 4:  vx = 0.0f; vy = -1.0f; vz = phi;  break;
                case 5:  vx = 0.0f; vy = 1.0f;  vz = phi;  break;
                case 6:  vx = 0.0f; vy = -1.0f; vz = -phi; break;
                case 7:  vx = 0.0f; vy = 1.0f;  vz = -phi; break;
                case 8:  vx = phi;  vy = 0.0f;  vz = -1.0f; break;
                case 9:  vx = phi;  vy = 0.0f;  vz = 1.0f;  break;
                case 10: vx = -phi; vy = 0.0f;  vz = -1.0f; break;
                case 11: vx = -phi; vy = 0.0f;  vz = 1.0f;  break;
                default: break;
            }
            float norm = sqrtf(vx * vx + vy * vy + vz * vz);
            if (norm > 0.0f) {
                vx /= norm;
                vy /= norm;
                vz /= norm;
            }
            break;
        }
        default: { // general circle / fallback
            float angle = (float)tid * (2.0f * 3.14159265358979323846f / max(1u, vertex_count));
            vx = cosf(angle);
            vy = sinf(angle);
            vz = 0.0f;
            break;
        }
    }

    unsigned int offset = tid * 3;
    vertices[offset + 0] = vx * scale;
    vertices[offset + 1] = vy * scale;
    vertices[offset + 2] = vz * scale;
}
