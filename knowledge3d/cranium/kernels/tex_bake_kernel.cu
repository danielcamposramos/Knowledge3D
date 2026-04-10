constexpr uint32_t kTextureSlotCount = 256u;
constexpr uint32_t kTextureBakedCount = 64u;
constexpr uint32_t kInvalidTextureHandle = 0u;
constexpr uint32_t kInvalidTextureSlot = 0xffffffffu;
constexpr uint32_t kNoTextureId = 0xffffffffu;

__device__ inline TextureHandlePool* texture_pool() {
    return reinterpret_cast<TextureHandlePool*>(g_texture_pool_ptr);
}

__device__ inline uint32_t texture_handle_to_slot(float handle_scalar) {
    const int handle = static_cast<int>(floorf(handle_scalar + 0.5f));
    if (handle <= 0 || handle > static_cast<int>(kTextureSlotCount)) {
        return kInvalidTextureSlot;
    }
    return static_cast<uint32_t>(handle - 1);
}

__device__ inline float texture_slot_to_handle(uint32_t slot) {
    return static_cast<float>(slot + 1u);
}

__device__ inline bool texture_slot_valid(TextureHandlePool* pool, uint32_t slot) {
    return pool != nullptr &&
        slot < kTextureSlotCount &&
        pool->slot_ptr[slot] != 0ULL &&
        pool->width[slot] > 0u &&
        pool->height[slot] > 0u;
}

__device__ inline float* texture_slot_ptr(TextureHandlePool* pool, uint32_t slot) {
    if (!texture_slot_valid(pool, slot)) {
        return nullptr;
    }
    return reinterpret_cast<float*>(pool->slot_ptr[slot]);
}

__device__ inline uint32_t texture_alloc_slot(TextureHandlePool* pool) {
    if (pool == nullptr) {
        return kInvalidTextureHandle;
    }
    for (uint32_t slot = 0; slot < kTextureSlotCount; ++slot) {
        if (pool->slot_ptr[slot] == 0ULL) {
            continue;
        }
        if (pool->in_use[slot] == 0u) {
            pool->in_use[slot] = 1u;
            return slot + 1u;
        }
    }
    return kInvalidTextureHandle;
}

__device__ inline uint32_t texture_alloc_baked(TextureHandlePool* pool) {
    if (pool == nullptr) {
        return kInvalidTextureHandle;
    }
    for (uint32_t idx = 0; idx < kTextureBakedCount; ++idx) {
        if (pool->baked_in_use[idx] == 0u) {
            pool->baked_in_use[idx] = 1u;
            return idx + 1u;
        }
    }
    return kInvalidTextureHandle;
}

__device__ inline void texture_clear_slot(TextureHandlePool* pool, uint32_t slot) {
    if (!texture_slot_valid(pool, slot)) {
        return;
    }
    float* ptr = texture_slot_ptr(pool, slot);
    const uint32_t count = pool->width[slot] * pool->height[slot];
    for (uint32_t idx = 0; idx < count; ++idx) {
        ptr[idx] = 0.0f;
    }
}

__device__ inline uint32_t texture_bake_handle(TextureHandlePool* pool, float handle_scalar, uint32_t width, uint32_t height) {
    const uint32_t slot = texture_handle_to_slot(handle_scalar);
    if (!texture_slot_valid(pool, slot)) {
        return kInvalidTextureHandle;
    }
    if (width > 0u) {
        pool->width[slot] = width;
    }
    if (height > 0u) {
        pool->height[slot] = height;
    }
    const uint32_t baked_id = texture_alloc_baked(pool);
    if (baked_id == kInvalidTextureHandle) {
        return kInvalidTextureHandle;
    }
    pool->baked_source_slot[baked_id - 1u] = slot;
    return baked_id;
}
