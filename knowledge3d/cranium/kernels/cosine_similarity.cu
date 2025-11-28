// Batch cosine similarity kernel (sovereign PTX, no external deps).

extern "C" __global__ void cosine_similarity_batch(
    const float* candidates,  // [N, D]
    const float* expected,    // [D]
    float* scores,            // [N]
    int N,
    int D) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= N) return;

  float dot = 0.0f;
  float norm_c = 0.0f;
  for (int d = 0; d < D; ++d) {
    float v = candidates[idx * D + d];
    dot += v * expected[d];
    norm_c += v * v;
  }
  norm_c = sqrtf(norm_c);
  scores[idx] = (norm_c > 1e-8f) ? (dot / norm_c) : 0.0f;
}

extern "C" __global__ void compute_norm(const float* vec, float* norm_out, int D) {
  extern __shared__ float shared[];
  int tid = threadIdx.x;
  float sum = 0.0f;
  for (int d = tid; d < D; d += blockDim.x) {
    float v = vec[d];
    sum += v * v;
  }
  shared[tid] = sum;
  __syncthreads();
  for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
    if (tid < stride) shared[tid] += shared[tid + stride];
    __syncthreads();
  }
  if (tid == 0) *norm_out = sqrtf(shared[0]);
}
