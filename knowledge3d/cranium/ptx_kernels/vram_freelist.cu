/*
 * vram_freelist.cu — Device-side slab allocator for Galaxy star creation
 *
 * Spec: CLAUDE_CODEX_GPU_GAME_LOOP_CLOSURE_04.18.2026.md §2.1, §2 (VramFreelist)
 * Audit: AGENT1_KERNEL_AUDIT_04.18.2026.md §6 (multi-hop double-buffer note)
 *
 * WHY a pre-allocated free-list instead of cudaMalloc on the hot path:
 *   cudaMalloc is a host-side API call — it cannot be called from inside a
 *   running kernel. Any Galaxy star that must be CREATED during a tick (new
 *   inference result, crystallized pattern, synthesized symbol) needs a slot
 *   from a pre-allocated pool. This free-list is populated once at boot by
 *   vram_freelist_init() and thereafter managed entirely on-device.
 *
 * Allocation strategy:
 *   A lock-free stack implemented as a VRAM-resident uint32_t array
 *   (free_list) plus a head index. alloc = pop from top; release = push to top.
 *   Uses atomicCAS spin-loop for correctness under concurrent pops.
 *   Each slot is `slab_bytes` bytes; slab_memory is a contiguous VRAM region.
 *
 * Thread-safety: safe for concurrent calls from different GPU threads.
 *   Multiple threads allocating simultaneously will CAS-compete; only one wins
 *   per iteration. Under low contention (star creation is rare mid-tick) this
 *   is fine. Heavy creation should happen in sleep-time with less contention.
 *
 * Overflow handling:
 *   alloc returns K3D_FREELIST_INVALID_IDX (0xFFFFFFFFu) when exhausted.
 *   Callers must check and write tick_status = K3D_TICK_STATUS_FREELIST_OOM.
 *   No fallback to cudaMalloc. EVER.
 *
 * Target: sm_86 (RTX 3070). No host includes.
 */

#include <cuda_runtime.h>
#include <cstdint>

/* Sentinel value: returned by freelist_alloc when the pool is empty */
#define K3D_FREELIST_INVALID_IDX 0xFFFFFFFFu

/*
 * VramFreelist — opaque descriptor for a device-managed slab pool.
 *
 * Fields:
 *   free_list      — array of free slot indices, managed as a lock-free stack
 *   free_list_head — index into free_list[] pointing to the current top-of-stack
 *   slab_base_idx  — unused field (reserved; base index always 0)
 *   capacity       — total number of slots
 *   slab_bytes     — size in bytes of each slot
 *   slab_memory    — flat VRAM buffer: slot[i] starts at slab_memory + i*slab_bytes
 *
 * NOTE: All pointers must be device-accessible (cudaMalloc'd by host at boot).
 */
struct VramFreelist {
    uint32_t* free_list;       /* device ptr: stack of free slot indices [capacity] */
    uint32_t* free_list_head;  /* device ptr: single uint32, index of stack top     */
    uint32_t* slab_base_idx;   /* reserved, may be nullptr                           */
    uint32_t  capacity;        /* total slot count                                   */
    uint32_t  slab_bytes;      /* bytes per slot                                     */
    void*     slab_memory;     /* device ptr: flat slab region capacity*slab_bytes   */
};

/* ---------------------------------------------------------------------------
 * freelist_alloc — pop one slot index from the free-list stack.
 *
 * Returns the slot index (< capacity), or K3D_FREELIST_INVALID_IDX if empty.
 * Uses a CAS spin loop so concurrent threads safely compete.
 * Caller responsibility: check for INVALID before using the index.
 * --------------------------------------------------------------------------- */
__device__ uint32_t freelist_alloc(VramFreelist* fl)
{
    if (fl == nullptr) return K3D_FREELIST_INVALID_IDX;

    uint32_t old_head, new_head, slot_idx;
    do {
        old_head = atomicAdd(fl->free_list_head, 0u); /* volatile read */
        if (old_head == 0u) {
            /* Stack empty: no available slots */
            return K3D_FREELIST_INVALID_IDX;
        }
        /*
         * Stack is 1-indexed: free_list_head == N means N entries available,
         * with the top entry at free_list[N-1].
         */
        slot_idx = fl->free_list[old_head - 1u];
        new_head = old_head - 1u;
    } while (atomicCAS(fl->free_list_head, old_head, new_head) != old_head);

    /* Fence before the caller writes into the slot so prior writes are ordered */
    __threadfence();
    return slot_idx;
}

/* ---------------------------------------------------------------------------
 * freelist_release — push a slot index back onto the free-list stack.
 *
 * idx must be a value previously returned by freelist_alloc.
 * Safe to call from multiple threads concurrently.
 * --------------------------------------------------------------------------- */
__device__ void freelist_release(VramFreelist* fl, uint32_t idx)
{
    if (fl == nullptr || idx >= fl->capacity) return;

    uint32_t old_head, new_head;
    do {
        old_head = atomicAdd(fl->free_list_head, 0u); /* volatile read */
        if (old_head >= fl->capacity) {
            /* Stack already full: this would indicate a double-free bug */
            return;
        }
        /* Fence to ensure the slot payload is visible before it is reusable */
        __threadfence();
        fl->free_list[old_head] = idx;
        new_head = old_head + 1u;
    } while (atomicCAS(fl->free_list_head, old_head, new_head) != old_head);
}

/* ---------------------------------------------------------------------------
 * freelist_slot_ptr — compute byte pointer to slot i's slab region.
 *
 * Returns nullptr if fl is null or idx >= capacity.
 * The returned pointer is valid for fl->slab_bytes bytes.
 * --------------------------------------------------------------------------- */
__device__ void* freelist_slot_ptr(VramFreelist* fl, uint32_t idx)
{
    if (fl == nullptr || idx >= fl->capacity || fl->slab_memory == nullptr) {
        return nullptr;
    }
    return static_cast<uint8_t*>(fl->slab_memory) +
           static_cast<uint64_t>(idx) * static_cast<uint64_t>(fl->slab_bytes);
}

/* ---------------------------------------------------------------------------
 * vram_freelist_init — host-side boot helper.
 *
 * Called ONCE from Python/C++ boot code (NOT from any kernel).
 * Allocates all device memory and initialises the free-list stack so every
 * slot is available (stack is full: free_list[i] = i, head = capacity).
 *
 * Parameters:
 *   fl          — pointer to a host-resident VramFreelist descriptor struct
 *   capacity    — number of slots to pre-allocate
 *   slab_bytes  — size in bytes of each slot (must be multiple of 16 for alignment)
 *
 * The caller owns the VramFreelist struct (may be stack-allocated on host).
 * After this call, fl->free_list, fl->free_list_head, fl->slab_memory are all
 * device pointers; pass fl to any kernel that needs it (by value or via constant
 * memory).
 * --------------------------------------------------------------------------- */
void vram_freelist_init(VramFreelist* fl, uint32_t capacity, uint32_t slab_bytes)
{
    if (fl == nullptr || capacity == 0u || slab_bytes == 0u) return;

    fl->capacity   = capacity;
    fl->slab_bytes = slab_bytes;

    /* Allocate the free-list index array */
    cudaMalloc(reinterpret_cast<void**>(&fl->free_list),
               static_cast<size_t>(capacity) * sizeof(uint32_t));

    /* Allocate the head counter (single uint32) */
    cudaMalloc(reinterpret_cast<void**>(&fl->free_list_head), sizeof(uint32_t));

    /* Allocate the slab memory region */
    cudaMalloc(&fl->slab_memory,
               static_cast<size_t>(capacity) * static_cast<size_t>(slab_bytes));

    fl->slab_base_idx = nullptr; /* reserved */

    /*
     * Initialise: fill free_list[i] = i for all i, then set head = capacity.
     * Use a small host-side loop + cudaMemcpy; this is boot-time, not hot-path.
     */
    uint32_t* host_init = new uint32_t[capacity];
    for (uint32_t i = 0u; i < capacity; ++i) {
        host_init[i] = i;
    }
    cudaMemcpy(fl->free_list, host_init,
               static_cast<size_t>(capacity) * sizeof(uint32_t),
               cudaMemcpyHostToDevice);
    delete[] host_init;

    /* Set head = capacity: all slots available */
    cudaMemcpy(fl->free_list_head, &capacity, sizeof(uint32_t),
               cudaMemcpyHostToDevice);
}
