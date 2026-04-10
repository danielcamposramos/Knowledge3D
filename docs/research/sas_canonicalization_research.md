**1. CANONICAL ORDERING RULES**

Define a total order `<` on STAR nodes:

*Tier 1 – Class:* `CONST` (0) `<` `SYMBOL` (1) `<` `EXPR` (2).  
*Tier 2 – Value:*  
• `CONST`: sort by IEEE-754 `uint64` bit-pattern (handles NaN/uniformly).  
• `SYMBOL`: sort by `symbol_id` (u32).  
• `EXPR`: sort by `(opcode, arity, canonical_hash(child_list))`, where `canonical_hash` is a 64-bit xxHash3 of the opcode and the sorted child hashes.

*Commutative Ops:* For `OP_ADD` and `OP_MUL`, sort arguments ascending by the tuple `(class, value/canonical_hash)`.  
*Non-commutative:* `OP_POW` keeps `(base, exp)` fixed; `OP_SUB` and `OP_DIV` are treated as non-commutative.

*Nested Expressions:* Comparison of two `EXPR` nodes is lexicographic on their sorted operand lists using pre-computed canonical subtree hashes (cached in the node’s `aux_hash` field).

**2. NORMALIZATION PASSES (Fused Bottom-Up)**

Single kernel pass per DAG node:

1. **Constant Folding:** If all children are `CONST`, replace node with `CONST(eval(op, children))`.  
2. **Identity/Annihilator:** `OP_MUL(x,1)→x`; `OP_MUL(x,0)→0`; `OP_ADD(x,0)→x`; `OP_POW(x,0)→1`; `OP_POW(x,1)→x`.  
3. **Flattening (Associativity):** `OP_ADD(a, OP_ADD(b,c))` becomes n-ary `OP_ADD(a,b,c)` stored as a right-deep spine with canonical sort applied to the flattened list.  
4. **Canonical Sort:** Apply the comparator from §1 to reorder children of commutative ops.

**3. HASHCONS ON GPU**

Structure: Global hash table `Slot { uint64_t key_hash; uint32_t node_idx; uint32_t lock; }`, size power-of-two (e.g., 2²⁴).  
Key encoding (128-bit): `{ opcode : 8; child_idx[0] : 32; child_idx[1] : 32; flags : 8; padding }`.

*Insertion Protocol (PTX):*
```
// Per-thread
uint64_t h = hash(key);
uint32_t slot = h & (TABLE_SIZE-1);
while (true) {
    if (table[slot].key_hash == h && table[slot].node_idx != 0) {
        if (match_key(table[slot].node_idx, key)) 
            return table[slot].node_idx; // Deduplicated
    }
    if (table[slot].lock == 0 && atomicCAS(&table[slot].lock, 0, 1) == 0) {
        table[slot].key_hash = h;
        table[slot].node_idx = new_node_idx;
        return new_node_idx;
    }
    slot = (slot + 1) & (TABLE_SIZE-1); // Linear probe
}
```
*Warp Collaboration:* Before global lookup, broadcast keys via `__match_any_sync`. If any lane owns the key, use `__shfl_sync` to obtain the existing `node_idx`, avoiding atomic contention.

**4. OPCODE 0x238 OP_CANONICALIZE**

*Stack Effect:* Pop `star_handle` (u32 pool index); Push `canonical_handle` (u32).  
*Behavior:* Executes the fused normalization passes (§2) with hashcons (§3). The node pool is updated in-place: non-canonical nodes are replaced by forwarding pointers to canonical entries.  
*Output:* The canonical STAR DAG remains in the pool; the opcode does **not** emit an RPN string. SHA-256 hashing for `star_id` occurs in a subsequent `OP_HASH` step that serializes the now-canonical DAG.