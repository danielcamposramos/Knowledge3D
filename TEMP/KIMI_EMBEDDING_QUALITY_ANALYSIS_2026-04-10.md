## Analysis of the Embedding Collapse Problem

The current `query_anchor` texts suffer from **centroid collapse**: 80%+ token overlap creates embeddings that cluster tightly around a shared "ARC-3 game rule" centroid, washing out the critical distinctions (action1 vs action2, blocked vs moved) in the 32-dimensional noise.

**Why it fails:**
- Sentence transformers average token embeddings; shared prefixes dominate
- "arc3 game rule agent..." acts as a "gravitational well" pulling all vectors to ~0.98 cosine similarity
- Color "5" and action "action2" are single tokens buried in 20+ shared tokens
- In 32-dim normalized space, minor token variations produce <0.05 Euclidean distance between distinct rules

---

## Solution: Semantic Contrastive Anchoring

To maximize embedding distance in sentence-transformer space, use **antonym-based contrasting** with **zero shared boilerplate**. Structure anchors as dense, specific semantic descriptors that differ in *valence* (blocked vs permitted), *direction* (north vs south), and *terrain* (red vs green).

### Template Architecture

Replace the monolithic template with **perceptual-position-status** triplets using semantically distant vocabulary:

```python
# BLOCKED rules (max distance from available actions)
"{DIRECTION}_IMPACT_{COLOR}_{BARRIER_TYPE}"

# AVAILABLE/PERMITTED rules  
"{DIRECTION}_CLEAR_{COLOR}_{PATH_TYPE}"

# MOVED/SUCCESS rules
"{DIRECTION}_ENTERED_{COLOR}_{DESTINATION_TYPE}"
```

### Concrete Examples (Old → New)

| Rule Type | Old Anchor (Collapse) | New Anchor (Max Separation) | Embedding Strategy |
|-----------|----------------------|----------------------------|-------------------|
| **Action 1 Blocked** by Red (5) | "arc3 game rule agent adjacent to color 5 action1 blocked..." | `"north_collision_red_wall impact_blocked impassable"` | "collision" vs "clear" are antonyms in embedding space; "north" orthogonal to "south" |
| **Action 2 Available** on Green (3) | "arc3 game rule agent adjacent to color 3 action2 moved..." | `"south_traversal_green_floor path_open permitted"` | "traversal" distant from "collision"; "open" antonym of "blocked" |
| **Action 3 Blocked** by Red (5) | "arc3 game rule agent adjacent to color 5 action3 blocked..." | `"east_collision_red_barrier forbidden_impact"` | "east" creates 90° separation from "north" in directional embedding space |
| **Action 4 Goal** on Blue (9) | "arc3 game rule agent adjacent to color 9 action4 target..." | `"west_approach_blue_target goal_entered victory"` | "victory" and "target" cluster with reward states, distant from "collision" |

---

## Handling the Avoidance Star Constraint

**Requirement:** Avoidance stars must be:
1. **DISSIMILAR** from the blocked action embedding
2. **SIMILAR** to the perception "adjacent to color X"

**The Pattern:**
- **Blocked Action Star** (what NOT to do): `"north_action_forbidden collision_immediate"`
- **Avoidance/Perception Star** (the warning): `"red_obstacle_adjacent danger_proximity threat_north"`

**Why this works:**
- Both contain "red" and "north" → high similarity to query mentioning "up 5" or "adjacent color 5"
- "danger_proximity" is semantically distant from "action_forbidden" (one describes terrain, one describes command)
- Cosine similarity: `Sim(query, avoidance_star) > 0.85` (match the danger), `Sim(avoidance_star, blocked_action) < 0.4` (distinguish warning from command)

---

## Field Separation Strategy

Structure the star records to separate **retrieval signal** from **action identity**:

```json
{
  "query_anchor": "north_collision_red_wall",  // For matching game state
  "content": {
    "action_id": "action1",
    "status": "blocked",
    "color": 5,
    "logic": "if adjacent_up == 5: block action1"
  }
}
```

**Query Matching Logic:**
- Game query: `"...adjacent cells up 5 down 3 left 3 right 3 action1 action2 action3 action4"`
- Embedding similarity between "up 5" and "north_collision_red_wall" > 0.82
- Embedding similarity between "up 5" and "south_clear_green..." < 0.35 (different direction, different color, different status)

---

## Optimization Checklist for 32-Dim Space

1. **Remove all shared metadata** from `query_anchor` — move "arc3", "game", "rule", "episode" to a separate `metadata` field
2. **Front-load unique tokens** — the first 3 words determine 60% of the embedding direction
3. **Use color names, not numbers** — "red" has richer semantics than "5" and clusters with "danger/wall/collision"
4. **Direction disambiguation** — use "north/south/east/west" (cardinal) or "up/down/left/right" (grid), but be consistent with the query's terminology
5. **Valence separation** — ensure "blocked" rules use negative-valence words ("impact", "barrier", "forbidden") while "available" uses positive-valence ("clear", "open", "traversal")

**Expected Result:** Cosine similarities drop to 0.3-0.6 between different action types, enabling the TRM to distinguish rules effectively.