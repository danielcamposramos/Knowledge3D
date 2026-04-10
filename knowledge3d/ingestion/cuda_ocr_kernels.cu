/* CUDA kernels for GPU-accelerated OCR preprocessing
 * 
 * These kernels provide high-performance image preprocessing for OCR operations,
 * including contrast enhancement, brightness adjustment, gamma correction,
 * and text region detection.
 */

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>
#include <stdio.h>

#define BLOCK_SIZE 16
#define MAX_THREADS_PER_BLOCK 256

/* Image preprocessing kernel for OCR
 * Applies contrast, brightness, and gamma correction
 */
extern "C" __global__ void preprocess_image_for_ocr(
    const unsigned char* __restrict__ input,
    unsigned char* __restrict__ output,
    int width,
    int height,
    float contrast,
    float brightness,
    float gamma
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        int idx = y * width + x;
        float pixel = (float)input[idx];
        
        // Apply contrast and brightness adjustments
        pixel = (pixel - 128.0f) * contrast + 128.0f + brightness;
        pixel = fmaxf(0.0f, fminf(255.0f, pixel));
        
        // Apply gamma correction
        pixel = 255.0f * powf(pixel / 255.0f, gamma);
        
        output[idx] = (unsigned char)pixel;
    }
}

/* Sobel edge detection kernel for text region identification
 * Calculates gradient magnitude to identify potential text areas
 */
extern "C" __global__ void sobel_edge_detection(
    const unsigned char* __restrict__ input,
    float* __restrict__ gradient_magnitude,
    int width,
    int height,
    float threshold
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x > 0 && x < width - 1 && y > 0 && y < height - 1) {
        int idx = y * width + x;
        
        // Sobel operators for gradient calculation
        float gx = -input[idx - width - 1] + input[idx - width + 1]
                  -2.0f * input[idx - 1] + 2.0f * input[idx + 1]
                  - input[idx + width - 1] + input[idx + width + 1];
        
        float gy = -input[idx - width - 1] - 2.0f * input[idx - width] - input[idx - width + 1]
                  + input[idx + width - 1] + 2.0f * input[idx + width] + input[idx + width + 1];
        
        float magnitude = sqrtf(gx * gx + gy * gy);
        
        // Apply threshold and normalize
        gradient_magnitude[idx] = magnitude > threshold ? magnitude / 255.0f : 0.0f;
    } else if (x < width && y < height) {
        int idx = y * width + x;
        gradient_magnitude[idx] = 0.0f;
    }
}

/* Adaptive threshold kernel for text binarization
 * Applies local adaptive thresholding to enhance text visibility
 */
extern "C" __global__ void adaptive_threshold(
    const unsigned char* __restrict__ input,
    unsigned char* __restrict__ output,
    int width,
    int height,
    int window_size,
    float offset
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        int idx = y * width + x;
        
        // Calculate local mean in window
        float sum = 0.0f;
        int count = 0;
        int half_window = window_size / 2;
        
        for (int dy = -half_window; dy <= half_window; dy++) {
            for (int dx = -half_window; dx <= half_window; dx++) {
                int nx = x + dx;
                int ny = y + dy;
                
                if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                    sum += (float)input[ny * width + nx];
                    count++;
                }
            }
        }
        
        float local_mean = sum / (float)count;
        float threshold = local_mean - offset;
        
        // Apply threshold
        output[idx] = (input[idx] > (unsigned char)threshold) ? 255 : 0;
    }
}

/* Morphological dilation kernel for text enhancement
 * Expands text regions to improve connectivity
 */
extern "C" __global__ void morphological_dilation(
    const unsigned char* __restrict__ input,
    unsigned char* __restrict__ output,
    int width,
    int height,
    int dilation_size
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        int idx = y * width + x;
        unsigned char max_val = 0;
        
        int half_size = dilation_size / 2;
        
        // Find maximum in dilation window
        for (int dy = -half_size; dy <= half_size; dy++) {
            for (int dx = -half_size; dx <= half_size; dx++) {
                int nx = x + dx;
                int ny = y + dy;
                
                if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                    max_val = fmaxf(max_val, input[ny * width + nx]);
                }
            }
        }
        
        output[idx] = max_val;
    }
}

/* Connected component labeling kernel for text region segmentation
 * Groups connected pixels into components for text region identification
 */
extern "C" __global__ void connected_components_init(
    int* __restrict__ labels,
    const unsigned char* __restrict__ binary_image,
    int width,
    int height
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        int idx = y * width + x;
        
        if (binary_image[idx] > 0) {
            labels[idx] = idx;  // Initialize with pixel index
        } else {
            labels[idx] = -1;   // Background
        }
    }
}

/* Union-find kernel for connected component merging
 * Merges connected components using union-find algorithm
 */
extern "C" __global__ void union_find_merge(
    int* __restrict__ labels,
    const unsigned char* __restrict__ binary_image,
    int width,
    int height
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        int idx = y * width + x;
        
        if (binary_image[idx] > 0) {
            // Check 4-connected neighbors
            int neighbors[4] = {
                (y > 0) ? (idx - width) : -1,
                (x > 0) ? (idx - 1) : -1,
                (x < width - 1) ? (idx + 1) : -1,
                (y < height - 1) ? (idx + width) : -1
            };
            
            int root = labels[idx];
            if (root != -1) {
                // Find minimum label among connected neighbors
                for (int i = 0; i < 4; i++) {
                    if (neighbors[i] != -1 && labels[neighbors[i]] != -1) {
                        int neighbor_root = labels[neighbors[i]];
                        if (neighbor_root < root) {
                            root = neighbor_root;
                        }
                    }
                }
                
                // Update label if smaller root found
                if (root < labels[idx]) {
                    labels[idx] = root;
                }
            }
        }
    }
}

/* Texture analysis kernel for text quality assessment
 * Analyzes local texture properties to assess text quality
 */
extern "C" __global__ void texture_analysis(
    const unsigned char* __restrict__ input,
    float* __restrict__ texture_score,
    int width,
    int height,
    int window_size
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        int idx = y * width + x;
        
        if (window_size <= 1) {
            texture_score[idx] = 0.0f;
            return;
        }
        
        float mean = 0.0f;
        float variance = 0.0f;
        int half_window = window_size / 2;
        int count = 0;
        
        // Calculate local mean
        for (int dy = -half_window; dy <= half_window; dy++) {
            for (int dx = -half_window; dx <= half_window; dx++) {
                int nx = x + dx;
                int ny = y + dy;
                
                if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                    mean += (float)input[ny * width + nx];
                    count++;
                }
            }
        }
        
        if (count > 0) {
            mean /= (float)count;
            
            // Calculate local variance
            for (int dy = -half_window; dy <= half_window; dy++) {
                for (int dx = -half_window; dx <= half_window; dx++) {
                    int nx = x + dx;
                    int ny = y + dy;
                    
                    if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                        float diff = (float)input[ny * width + nx] - mean;
                        variance += diff * diff;
                    }
                }
            }
            
            variance /= (float)count;
            
            // Texture score based on variance (higher variance = more texture = likely text)
            texture_score[idx] = fminf(1.0f, variance / 5000.0f);  // Normalize
        } else {
            texture_score[idx] = 0.0f;
        }
    }
}

/* Noise reduction kernel using median filter
 * Reduces noise while preserving edges
 */
extern "C" __global__ void median_filter(
    const unsigned char* __restrict__ input,
    unsigned char* __restrict__ output,
    int width,
    int height,
    int filter_size
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        int idx = y * width + x;
        int half_size = filter_size / 2;
        
        // Collect neighborhood pixels
        #define MAX_FILTER_SIZE 9
        unsigned char values[MAX_FILTER_SIZE * MAX_FILTER_SIZE];
        int count = 0;
        
        for (int dy = -half_size; dy <= half_size; dy++) {
            for (int dx = -half_size; dx <= half_size; dx++) {
                int nx = x + dx;
                int ny = y + dy;
                
                if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                    values[count++] = input[ny * width + nx];
                }
            }
        }
        
        // Simple bubble sort for median (small neighborhood)
        for (int i = 0; i < count - 1; i++) {
            for (int j = 0; j < count - i - 1; j++) {
                if (values[j] > values[j + 1]) {
                    unsigned char temp = values[j];
                    values[j] = values[j + 1];
                    values[j + 1] = temp;
                }
            }
        }
        
        // Take median value
        output[idx] = values[count / 2];
    }
}

/* Helper function to calculate optimal block dimensions */
dim3 calculateGridDim(int width, int height) {
    dim3 gridDim(
        (width + BLOCK_SIZE - 1) / BLOCK_SIZE,
        (height + BLOCK_SIZE - 1) / BLOCK_SIZE,
        1
    );
    return gridDim;
}

/* Helper function to calculate optimal thread dimensions */
dim3 calculateBlockDim(int width, int height) {
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE, 1);
    return blockDim;
}

/* Memory allocation helper for GPU processing */
cudaError_t allocateOCRBuffers(
    unsigned char** d_input,
    unsigned char** d_output,
    float** d_gradient,
    int** d_labels,
    float** d_texture,
    int width,
    int height
) {
    size_t image_size = width * height * sizeof(unsigned char);
    size_t float_size = width * height * sizeof(float);
    size_t int_size = width * height * sizeof(int);
    
    cudaError_t err;
    
    err = cudaMalloc(d_input, image_size);
    if (err != cudaSuccess) return err;
    
    err = cudaMalloc(d_output, image_size);
    if (err != cudaSuccess) return err;
    
    err = cudaMalloc(d_gradient, float_size);
    if (err != cudaSuccess) return err;
    
    err = cudaMalloc(d_labels, int_size);
    if (err != cudaSuccess) return err;
    
    err = cudaMalloc(d_texture, float_size);
    if (err != cudaSuccess) return err;
    
    return cudaSuccess;
}

/* Cleanup helper for GPU buffers */
void cleanupOCRBuffers(
    unsigned char* d_input,
    unsigned char* d_output,
    float* d_gradient,
    int* d_labels,
    float* d_texture
) {
    if (d_input) cudaFree(d_input);
    if (d_output) cudaFree(d_output);
    if (d_gradient) cudaFree(d_gradient);
    if (d_labels) cudaFree(d_labels);
    if (d_texture) cudaFree(d_texture);
}

/* Main OCR preprocessing pipeline kernel launcher */
extern "C" void launch_ocr_preprocessing_pipeline(
    const unsigned char* h_input,
    unsigned char* h_output,
    int width,
    int height,
    float contrast,
    float brightness,
    float gamma,
    float edge_threshold,
    int window_size,
    float offset,
    int dilation_size,
    int median_size
) {
    // Allocate device memory
    unsigned char *d_input, *d_output, *d_temp;
    float *d_gradient, *d_texture;
    int *d_labels;
    
    cudaError_t err = allocateOCRBuffers(
        &d_input, &d_output, &d_gradient, &d_labels, &d_texture,
        width, height
    );
    
    if (err != cudaSuccess) {
        printf("Failed to allocate GPU memory: %s\n", cudaGetErrorString(err));
        return;
    }
    
    // Allocate temporary buffer
    err = cudaMalloc(&d_temp, width * height * sizeof(unsigned char));
    if (err != cudaSuccess) {
        printf("Failed to allocate temporary buffer: %s\n", cudaGetErrorString(err));
        cleanupOCRBuffers(d_input, d_output, d_gradient, d_labels, d_texture);
        return;
    }
    
    // Copy input to device
    size_t image_size = width * height * sizeof(unsigned char);
    err = cudaMemcpy(d_input, h_input, image_size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        printf("Failed to copy input to device: %s\n", cudaGetErrorString(err));
        cleanupOCRBuffers(d_input, d_output, d_gradient, d_labels, d_texture);
        cudaFree(d_temp);
        return;
    }
    
    // Calculate grid and block dimensions
    dim3 gridDim = calculateGridDim(width, height);
    dim3 blockDim = calculateBlockDim(width, height);
    
    // Step 1: Preprocess image (contrast, brightness, gamma)
    preprocess_image_for_ocr<<<gridDim, blockDim>>>(
        d_input, d_temp, width, height, contrast, brightness, gamma
    );
    cudaDeviceSynchronize();
    
    // Step 2: Apply median filter for noise reduction
    median_filter<<<gridDim, blockDim>>>(d_temp, d_output, width, height, median_size);
    cudaDeviceSynchronize();
    
    // Step 3: Sobel edge detection
    sobel_edge_detection<<<gridDim, blockDim>>>(d_output, d_gradient, width, height, edge_threshold);
    cudaDeviceSynchronize();
    
    // Step 4: Adaptive thresholding
    adaptive_threshold<<<gridDim, blockDim>>>(d_output, d_temp, width, height, window_size, offset);
    cudaDeviceSynchronize();
    
    // Step 5: Morphological dilation
    morphological_dilation<<<gridDim, blockDim>>>(d_temp, d_output, width, height, dilation_size);
    cudaDeviceSynchronize();
    
    // Step 6: Texture analysis
    texture_analysis<<<gridDim, blockDim>>>(d_output, d_texture, width, height, window_size);
    cudaDeviceSynchronize();
    
    // Copy result back to host
    err = cudaMemcpy(h_output, d_output, image_size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        printf("Failed to copy output to host: %s\n", cudaGetErrorString(err));
    }
    
    // Cleanup
    cleanupOCRBuffers(d_input, d_output, d_gradient, d_labels, d_texture);
    cudaFree(d_temp);
}