#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>

namespace {
constexpr int kStackCapacity = 64;
constexpr int kMatrixMaxDim = 3;

enum class ItemType : uint8_t {
    kScalar = 0,
    kVector = 1,
    kMatrixRow = 2,
};

constexpr uint32_t kErrorNone = 0;
constexpr uint32_t kErrorUnknownOpcode = 9001;
constexpr uint32_t kErrorStackUnderflow = 9002;
constexpr uint32_t kErrorStackOverflow = 9003;
constexpr uint32_t kErrorInvalidMatrixDims = 9013;
constexpr uint32_t kErrorSingularMatrix = 9014;

struct StackItem {
    float value[4];
    ItemType type;
    uint8_t rows;
    uint8_t cols;
    uint8_t row_index;
};

struct alignas(16) InstanceState {
    uint32_t head;
    uint32_t size;
    uint32_t error;
    uint32_t reserved;
    float stack[kStackCapacity][4];
};

static_assert(sizeof(InstanceState) == 1040, "InstanceState layout mismatch");

__device__ inline float pack_meta(ItemType type, uint8_t rows, uint8_t cols, uint8_t row_index) {
    uint32_t bits = static_cast<uint32_t>(type) |
                    (static_cast<uint32_t>(rows) << 8) |
                    (static_cast<uint32_t>(cols) << 16) |
                    (static_cast<uint32_t>(row_index) << 24);
    return __uint_as_float(bits);
}

__device__ inline bool push_item(StackItem* stack, uint32_t& size, const StackItem& item, uint32_t& error) {
    if (size >= kStackCapacity) {
        error = kErrorStackOverflow;
        return false;
    }
    stack[size] = item;
    size += 1;
    return true;
}

__device__ inline bool pop_item(StackItem* stack, uint32_t& size, StackItem& out, uint32_t& error) {
    if (size == 0) {
        error = kErrorStackUnderflow;
        return false;
    }
    size -= 1;
    out = stack[size];
    return true;
}

__device__ inline bool pop_scalar(StackItem* stack, uint32_t& size, float& value, uint32_t& error) {
    StackItem item{};
    if (!pop_item(stack, size, item, error)) {
        return false;
    }
    if (item.type != ItemType::kScalar) {
        error = kErrorUnknownOpcode;
        return false;
    }
    value = item.value[0];
    return true;
}

struct Matrix {
    int rows;
    int cols;
    float data[kMatrixMaxDim * kMatrixMaxDim];
};

__device__ inline bool pop_matrix(StackItem* stack, uint32_t& size, Matrix& matrix, uint32_t& error) {
    if (size == 0) {
        error = kErrorStackUnderflow;
        return false;
    }

    StackItem top = stack[size - 1];
    if (top.type != ItemType::kMatrixRow) {
        error = kErrorUnknownOpcode;
        return false;
    }

    const int rows = static_cast<int>(top.rows);
    const int cols = static_cast<int>(top.cols);

    if (rows <= 0 || cols <= 0 || rows > kMatrixMaxDim || cols > kMatrixMaxDim) {
        error = kErrorInvalidMatrixDims;
        return false;
    }
    if (size < static_cast<uint32_t>(rows)) {
        error = kErrorStackUnderflow;
        return false;
    }

    matrix.rows = rows;
    matrix.cols = cols;

    for (int r = rows - 1; r >= 0; --r) {
        StackItem row_item{};
        if (!pop_item(stack, size, row_item, error)) {
            return false;
        }
        if (row_item.type != ItemType::kMatrixRow ||
            row_item.rows != top.rows ||
            row_item.cols != top.cols) {
            error = kErrorUnknownOpcode;
            return false;
        }
        int base = r * cols;
        matrix.data[base + 0] = row_item.value[0];
        if (cols > 1) {
            matrix.data[base + 1] = row_item.value[1];
        }
        if (cols > 2) {
            matrix.data[base + 2] = row_item.value[2];
        }
    }
    return true;
}

__device__ inline bool push_matrix(StackItem* stack, uint32_t& size, const Matrix& matrix, uint32_t& error) {
    if (matrix.rows <= 0 || matrix.cols <= 0 ||
        matrix.rows > kMatrixMaxDim || matrix.cols > kMatrixMaxDim) {
        error = kErrorInvalidMatrixDims;
        return false;
    }
    if (size + static_cast<uint32_t>(matrix.rows) > kStackCapacity) {
        error = kErrorStackOverflow;
        return false;
    }

    for (int r = 0; r < matrix.rows; ++r) {
        StackItem item{};
        int base = r * matrix.cols;
        item.value[0] = matrix.data[base + 0];
        item.value[1] = (matrix.cols > 1) ? matrix.data[base + 1] : 0.0f;
        item.value[2] = (matrix.cols > 2) ? matrix.data[base + 2] : 0.0f;
        item.value[3] = 0.0f;
        item.type = ItemType::kMatrixRow;
        item.rows = static_cast<uint8_t>(matrix.rows);
        item.cols = static_cast<uint8_t>(matrix.cols);
        item.row_index = static_cast<uint8_t>(r);
        if (!push_item(stack, size, item, error)) {
            return false;
        }
    }
    return true;
}

__device__ inline bool push_scalar(StackItem* stack, uint32_t& size, float value, uint32_t& error) {
    StackItem item{};
    item.value[0] = value;
    item.value[1] = 0.0f;
    item.value[2] = 0.0f;
    item.value[3] = 0.0f;
    item.type = ItemType::kScalar;
    item.rows = 1;
    item.cols = 1;
    item.row_index = 0;
    return push_item(stack, size, item, error);
}

__device__ inline bool push_vector(StackItem* stack, uint32_t& size, const float* vectors, uint32_t index, uint32_t& error) {
    StackItem item{};
    if (vectors) {
        item.value[0] = vectors[index * 3 + 0];
        item.value[1] = vectors[index * 3 + 1];
        item.value[2] = vectors[index * 3 + 2];
    } else {
        item.value[0] = 0.0f;
        item.value[1] = 0.0f;
        item.value[2] = 0.0f;
    }
    item.value[3] = 0.0f;
    item.type = ItemType::kVector;
    item.rows = 1;
    item.cols = 3;
    item.row_index = 0;
    return push_item(stack, size, item, error);
}

__device__ inline float determinant_2x2(const Matrix& m) {
    return m.data[0] * m.data[3] - m.data[1] * m.data[2];
}

__device__ inline float determinant_3x3(const Matrix& m) {
    const float a = m.data[0];
    const float b = m.data[1];
    const float c = m.data[2];
    const float d = m.data[3];
    const float e = m.data[4];
    const float f = m.data[5];
    const float g = m.data[6];
    const float h = m.data[7];
    const float i = m.data[8];
    return a * (e * i - f * h) -
           b * (d * i - f * g) +
           c * (d * h - e * g);
}

__device__ inline bool inverse_2x2(const Matrix& input, Matrix& output) {
    const float det = determinant_2x2(input);
    if (fabsf(det) < 1e-8f) {
        return false;
    }
    const float inv_det = 1.0f / det;
    output.rows = 2;
    output.cols = 2;
    output.data[0] = input.data[3] * inv_det;
    output.data[1] = -input.data[1] * inv_det;
    output.data[2] = -input.data[2] * inv_det;
    output.data[3] = input.data[0] * inv_det;
    return true;
}

__device__ inline bool inverse_3x3(const Matrix& input, Matrix& output) {
    const float det = determinant_3x3(input);
    if (fabsf(det) < 1e-8f) {
        return false;
    }
    output.rows = 3;
    output.cols = 3;

    const float inv_det = 1.0f / det;

    output.data[0] = (input.data[4] * input.data[8] - input.data[5] * input.data[7]) * inv_det;
    output.data[1] = (input.data[2] * input.data[7] - input.data[1] * input.data[8]) * inv_det;
    output.data[2] = (input.data[1] * input.data[5] - input.data[2] * input.data[4]) * inv_det;

    output.data[3] = (input.data[5] * input.data[6] - input.data[3] * input.data[8]) * inv_det;
    output.data[4] = (input.data[0] * input.data[8] - input.data[2] * input.data[6]) * inv_det;
    output.data[5] = (input.data[2] * input.data[3] - input.data[0] * input.data[5]) * inv_det;

    output.data[6] = (input.data[3] * input.data[7] - input.data[4] * input.data[6]) * inv_det;
    output.data[7] = (input.data[1] * input.data[6] - input.data[0] * input.data[7]) * inv_det;
    output.data[8] = (input.data[0] * input.data[4] - input.data[1] * input.data[3]) * inv_det;

    return true;
}

__device__ inline void matmul(const Matrix& a, const Matrix& b, Matrix& out) {
    out.rows = a.rows;
    out.cols = b.cols;
    for (int r = 0; r < out.rows; ++r) {
        for (int c = 0; c < out.cols; ++c) {
            float acc = 0.0f;
            for (int k = 0; k < a.cols; ++k) {
                acc += a.data[r * a.cols + k] * b.data[k * b.cols + c];
            }
            out.data[r * out.cols + c] = acc;
        }
    }
}

__device__ inline void transpose(const Matrix& m, Matrix& out) {
    out.rows = m.cols;
    out.cols = m.rows;
    for (int r = 0; r < out.rows; ++r) {
        for (int c = 0; c < out.cols; ++c) {
            out.data[r * out.cols + c] = m.data[c * m.cols + r];
        }
    }
}
}  // namespace

extern "C" __global__ void modular_rpn_kernel_extended(
    uint32_t instance_id,
    const uint16_t* __restrict__ op_codes,
    const float* __restrict__ scalars,
    const float* __restrict__ vectors,
    const float* __restrict__ matrices,
    InstanceState* __restrict__ states,
    uint32_t token_count) {
    InstanceState* state = reinterpret_cast<InstanceState*>(
        reinterpret_cast<uint8_t*>(states) + instance_id * sizeof(InstanceState));

    StackItem stack[kStackCapacity];
    uint32_t stack_size = 0;
    uint32_t error_code = kErrorNone;

    uint32_t scalar_index = 0;
    uint32_t vector_index = 0;
    uint32_t matrix_index = 0;

    for (uint32_t i = 0; i < token_count && error_code == kErrorNone; ++i) {
        const uint16_t opcode = op_codes[i];

        switch (opcode) {
            case 0x00: {  // literal scalar
                float value = scalars ? scalars[scalar_index] : 0.0f;
                scalar_index += 1;
                push_scalar(stack, stack_size, value, error_code);
                break;
            }
            case 0x01: {  // literal vector
                push_vector(stack, stack_size, vectors, vector_index, error_code);
                vector_index += 1;
                break;
            }
            case 0x02: {  // literal matrix
                float rows_f = scalars ? scalars[scalar_index] : 0.0f;
                float cols_f = scalars ? scalars[scalar_index + 1] : 0.0f;
                scalar_index += 2;
                int rows = static_cast<int>(roundf(rows_f));
                int cols = static_cast<int>(roundf(cols_f));
                if (rows < 1) rows = 1;
                if (rows > kMatrixMaxDim) rows = kMatrixMaxDim;
                if (cols < 1) cols = 1;
                if (cols > kMatrixMaxDim) cols = kMatrixMaxDim;

                Matrix mat{};
                mat.rows = rows;
                mat.cols = cols;
                for (int r = 0; r < rows; ++r) {
                    for (int c = 0; c < cols; ++c) {
                        float value = matrices ? matrices[matrix_index] : 0.0f;
                        matrix_index += 1;
                        mat.data[r * cols + c] = value;
                    }
                }
                push_matrix(stack, stack_size, mat, error_code);
                break;
            }
            case 0x0A:  // add
            case 0x0B:  // sub
            case 0x0C:  // mul
            case 0x0D: {  // div
                float rhs = 0.0f;
                float lhs = 0.0f;
                if (!pop_scalar(stack, stack_size, rhs, error_code)) break;
                if (!pop_scalar(stack, stack_size, lhs, error_code)) break;
                float result = 0.0f;
                if (opcode == 0x0A) {
                    result = lhs + rhs;
                } else if (opcode == 0x0B) {
                    result = lhs - rhs;
                } else if (opcode == 0x0C) {
                    result = lhs * rhs;
                } else {
                    result = lhs / rhs;
                }
                push_scalar(stack, stack_size, result, error_code);
                break;
            }
            case 0x14:  // sqrt
            case 0x15:  // exp
            case 0x16:  // log
            case 0x18:  // sin
            case 0x19:  // cos
            case 0x1A: {  // tan
                float value = 0.0f;
                if (!pop_scalar(stack, stack_size, value, error_code)) break;
                float result = 0.0f;
                if (opcode == 0x14) {
                    result = sqrtf(value);
                } else if (opcode == 0x15) {
                    result = expf(value);
                } else if (opcode == 0x16) {
                    result = logf(value);
                } else if (opcode == 0x18) {
                    result = sinf(value);
                } else if (opcode == 0x19) {
                    result = cosf(value);
                } else {
                    result = tanf(value);
                }
                push_scalar(stack, stack_size, result, error_code);
                break;
            }
            case 0x28:  // gt
            case 0x2A:  // lt
            case 0x2C:  // eq
            case 0x2E:  // max
            case 0x2F: {  // min
                float rhs = 0.0f;
                float lhs = 0.0f;
                if (!pop_scalar(stack, stack_size, rhs, error_code)) break;
                if (!pop_scalar(stack, stack_size, lhs, error_code)) break;
                float result = 0.0f;
                if (opcode == 0x28) {
                    result = lhs > rhs ? 1.0f : 0.0f;
                } else if (opcode == 0x2A) {
                    result = lhs < rhs ? 1.0f : 0.0f;
                } else if (opcode == 0x2C) {
                    result = fabsf(lhs - rhs) < 1e-6f ? 1.0f : 0.0f;
                } else if (opcode == 0x2E) {
                    result = fmaxf(lhs, rhs);
                } else {
                    result = fminf(lhs, rhs);
                }
                push_scalar(stack, stack_size, result, error_code);
                break;
            }
            case 0x32: {  // dup
                if (stack_size == 0) {
                    error_code = kErrorStackUnderflow;
                    break;
                }
                StackItem top = stack[stack_size - 1];
                push_item(stack, stack_size, top, error_code);
                break;
            }
            case 0x33: {  // swap
                if (stack_size < 2) {
                    error_code = kErrorStackUnderflow;
                    break;
                }
                StackItem tmp = stack[stack_size - 1];
                stack[stack_size - 1] = stack[stack_size - 2];
                stack[stack_size - 2] = tmp;
                break;
            }
            case 0x34: {  // drop
                StackItem discarded{};
                pop_item(stack, stack_size, discarded, error_code);
                break;
            }
            case 0x5E: {  // trace (94 decimal)
                Matrix mat{};
                if (!pop_matrix(stack, stack_size, mat, error_code)) break;
                if (mat.rows != mat.cols) {
                    error_code = kErrorInvalidMatrixDims;
                    break;
                }
                float trace = 0.0f;
                for (int d = 0; d < mat.rows; ++d) {
                    trace += mat.data[d * mat.cols + d];
                }
                push_scalar(stack, stack_size, trace, error_code);
                break;
            }
            case 0x5A: {  // matmul (90 decimal)
                Matrix b{}, a{}, result{};
                if (!pop_matrix(stack, stack_size, b, error_code)) break;
                if (!pop_matrix(stack, stack_size, a, error_code)) break;
                if (a.cols != b.rows) {
                    error_code = kErrorInvalidMatrixDims;
                    break;
                }
                matmul(a, b, result);
                push_matrix(stack, stack_size, result, error_code);
                break;
            }
            case 0x5B: {  // transpose (91 decimal)
                Matrix input{}, result{};
                if (!pop_matrix(stack, stack_size, input, error_code)) break;
                transpose(input, result);
                push_matrix(stack, stack_size, result, error_code);
                break;
            }
            case 0x5C: {  // determinant (92 decimal)
                Matrix input{};
                if (!pop_matrix(stack, stack_size, input, error_code)) break;
                if (input.rows != input.cols) {
                    error_code = kErrorInvalidMatrixDims;
                    break;
                }
                float det = 0.0f;
                if (input.rows == 1) {
                    det = input.data[0];
                } else if (input.rows == 2) {
                    det = determinant_2x2(input);
                } else if (input.rows == 3) {
                    det = determinant_3x3(input);
                } else {
                    error_code = kErrorInvalidMatrixDims;
                    break;
                }
                push_scalar(stack, stack_size, det, error_code);
                break;
            }
            case 0x5D: {  // inverse (93 decimal)
                Matrix input{}, result{};
                if (!pop_matrix(stack, stack_size, input, error_code)) break;
                if (input.rows != input.cols) {
                    error_code = kErrorInvalidMatrixDims;
                    break;
                }
                bool ok = false;
                if (input.rows == 1) {
                    if (fabsf(input.data[0]) < 1e-8f) {
                        error_code = kErrorSingularMatrix;
                        break;
                    }
                    result.rows = 1;
                    result.cols = 1;
                    result.data[0] = 1.0f / input.data[0];
                    ok = true;
                } else if (input.rows == 2) {
                    ok = inverse_2x2(input, result);
                } else if (input.rows == 3) {
                    ok = inverse_3x3(input, result);
                }
                if (!ok) {
                    error_code = kErrorSingularMatrix;
                    break;
                }
                push_matrix(stack, stack_size, result, error_code);
                break;
            }
            default:
                error_code = kErrorUnknownOpcode;
                break;
        }
    }

    state->head = 0;
    state->size = stack_size;
    state->error = error_code;
    state->reserved = 0;

    const uint32_t count = (stack_size < kStackCapacity) ? stack_size : kStackCapacity;
    for (uint32_t idx = 0; idx < count; ++idx) {
        const StackItem& item = stack[idx];
        state->stack[idx][0] = item.value[0];
        state->stack[idx][1] = item.value[1];
        state->stack[idx][2] = item.value[2];
        state->stack[idx][3] = pack_meta(item.type, item.rows, item.cols, item.row_index);
    }
}
