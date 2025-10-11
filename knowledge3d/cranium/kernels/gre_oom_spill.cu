// OOM Spill Kernel
// Computes a spill plan based on available bytes and per-atom size.
// Parameters:
//   stats_ptr        : uint64[2]  (input: oldest_index, atom_size_bytes)
//   available_bytes  : uint64
//   request_count    : uint32     number of atoms requested for spill
//   out_ptr          : uint64[2]  (output: atoms_to_spill, bytes_required)

extern "C" __global__ void gre_oom_spill(
    const unsigned long long* stats_ptr,        // [oldest_index, atom_size_bytes]
    unsigned long long available_bytes,
    unsigned int request_count,
    unsigned long long* out_ptr                 // [atoms_to_spill, bytes_required]
)
{
    // Only thread 0 performs the work
    if (threadIdx.x != 0) return;

    // Load stats
    unsigned long long oldest_index = stats_ptr[0];
    unsigned long long atom_size = stats_ptr[1];

    // Determine maximum atoms that fit into available bytes
    unsigned long long max_atoms = 0;
    if (atom_size != 0) {
        max_atoms = available_bytes / atom_size;
    }

    // Take minimum of max_atoms and requested count
    unsigned long long atoms_to_spill = min(max_atoms, (unsigned long long)request_count);
    unsigned long long bytes_required = atoms_to_spill * atom_size;

    // Store results
    out_ptr[0] = atoms_to_spill;
    out_ptr[1] = bytes_required;
}
