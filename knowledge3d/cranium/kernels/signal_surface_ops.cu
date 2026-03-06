extern "C" __global__ void heightfield_to_vertices_kernel(
    const float* heightfield,
    float* vertices,
    int rows,
    int cols,
    float time_scale,
    float frequency_scale
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = rows * cols;
    if (idx >= total) {
        return;
    }

    int row = idx / cols;
    int col = idx - row * cols;

    float x = 0.0f;
    if (cols > 1) {
        x = (-0.5f + ((float)col / (float)(cols - 1))) * time_scale;
    }
    float z = 0.0f;
    if (rows > 1) {
        z = (0.5f - ((float)row / (float)(rows - 1))) * frequency_scale;
    }

    int out = idx * 3;
    vertices[out + 0] = x;
    vertices[out + 1] = heightfield[idx];
    vertices[out + 2] = z;
}

extern "C" __global__ void heightfield_to_normals_kernel(
    const float* heightfield,
    float* normals,
    int rows,
    int cols,
    float time_scale,
    float frequency_scale
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = rows * cols;
    if (idx >= total) {
        return;
    }

    int row = idx / cols;
    int col = idx - row * cols;

    int left_col = col > 0 ? col - 1 : col;
    int right_col = col + 1 < cols ? col + 1 : col;
    int up_row = row > 0 ? row - 1 : row;
    int down_row = row + 1 < rows ? row + 1 : row;

    float h_left = heightfield[row * cols + left_col];
    float h_right = heightfield[row * cols + right_col];
    float h_up = heightfield[up_row * cols + col];
    float h_down = heightfield[down_row * cols + col];

    float dx = cols > 1 ? time_scale / (float)(cols - 1) : 1.0f;
    float dz = rows > 1 ? frequency_scale / (float)(rows - 1) : 1.0f;

    float dx_span = (float)(right_col - left_col) * dx;
    if (dx_span < 1e-6f) {
        dx_span = 1.0f;
    }
    float dz_span = (float)(down_row - up_row) * dz;
    if (dz_span < 1e-6f) {
        dz_span = 1.0f;
    }

    float slope_x = (h_right - h_left) / dx_span;
    float slope_z = (h_up - h_down) / dz_span;

    float nx = -slope_x;
    float ny = 1.0f;
    float nz = -slope_z;
    float length = sqrtf(nx * nx + ny * ny + nz * nz);
    if (length < 1e-8f) {
        length = 1.0f;
    }

    int out = idx * 3;
    normals[out + 0] = nx / length;
    normals[out + 1] = ny / length;
    normals[out + 2] = nz / length;
}
