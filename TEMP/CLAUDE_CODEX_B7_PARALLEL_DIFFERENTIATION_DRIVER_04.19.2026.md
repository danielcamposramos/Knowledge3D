# Codex Spec — Parallel B7 Differentiation Driver (No Re-Work, No Overlap)

**Date**: 2026-04-19
**Owner**: Claude (spec author) — acting on Daniel's parallel-execution approval
**Severity**: P1 for B7 Step 3 throughput (3,083 clusters / 8,156 row-level cloud calls are not a same-turn batch)
**Scope**: Ingestion-path only. Extends `scripts/ingestion/d3/differentiate_b7_residual.py` with shard-claim + async-per-cluster execution. Zero hot-path impact. Zero changes to the proceduralizer contract, the web_search client, or the bundle validation — Steps 1-2 are done and stay intact.

---

## Daniel's constraint

> *"I agree on the parallel and adding async work as well, as long as each
> worker does not work on other work or do re-work."*

Two invariants this spec must guarantee:

1. **No overlap**: two workers never process the same cluster concurrently.
2. **No re-work**: a cluster that has already been resolved (including in a prior run) is not re-sent to the cloud model.

The design below gets both from a claim-based shard filesystem — no coordination daemon, no shared queue, no cross-worker RPC.

---

## Architecture — claim-based shards + per-cluster outputs

### Layout

```
scripts/ingestion/staging/D3_dedup/differentiate_b7/
├── clusters/                       # one file per cluster (input manifest)
│   ├── <content_hash>.cluster.json # { "content_hash", "row_ids": [...] }
│   └── ...
├── claims/                         # atomic ownership locks
│   ├── <content_hash>.claim        # worker_id + claimed_at; TTL 20 min
│   └── ...
├── done/                           # completed markers (source of truth for "no re-work")
│   ├── <content_hash>.done         # sha256 of enriched output + resolved_count
│   └── ...
├── enriched/                       # per-cluster enriched row outputs
│   ├── <content_hash>.jsonl        # enriched JSONL rows for this cluster
│   └── ...
├── unresolved/                     # per-cluster unresolved logs
│   ├── <content_hash>.jsonl
│   └── ...
└── workers/
    └── <worker_id>.log             # per-worker progress / error log
```

### Atomic claim protocol

Each worker runs this loop:

```
while unclaimed_cluster := pick_next_cluster():
    claim_path = claims/<content_hash>.claim
    try:
        # POSIX O_CREAT | O_EXCL is the atomicity primitive. If the file
        # already exists, open(...O_EXCL) raises FileExistsError and we
        # skip to the next cluster. Two workers racing on the same cluster
        # is impossible: exactly one creates the file.
        fd = os.open(claim_path, O_WRONLY | O_CREAT | O_EXCL, 0o644)
        os.write(fd, json.dumps({"worker_id": ..., "claimed_at": ...}))
        os.close(fd)
    except FileExistsError:
        # Someone else owns it. Check the claim's age:
        #   - if < 20 min: leave it alone, move on
        #   - if >= 20 min AND no done/<content_hash>.done: treat as
        #     crashed worker, steal by os.rename(claim_path, claim_path+".stale")
        #     then retry the O_EXCL create. Only ONE stealer wins.
        continue_or_steal()

    try:
        # Skip if already done in a prior run (no re-work)
        if exists(done/<content_hash>.done):
            release_claim()
            continue

        process_cluster(cluster)  # runs the async per-row pipeline
        write enriched/<content_hash>.jsonl
        write unresolved/<content_hash>.jsonl (if any)
        fsync both files
        atomically rename done/<content_hash>.done.tmp -> done/<content_hash>.done
    finally:
        release_claim()  # os.unlink(claim_path)
```

**Re-work protection**: the single `exists(done/<content_hash>.done)` check early in the loop is the invariant. Every resume, retry, or additional worker starts by reading that file. A cluster is processed exactly once globally.

**Overlap protection**: `O_CREAT | O_EXCL` is atomic at the filesystem syscall level. Two workers racing on the same cluster get exactly one success; the loser falls through to the next cluster.

### Cluster selection order

`pick_next_cluster()` iterates `clusters/*.cluster.json` in a deterministic order so different workers starting from scratch don't all hit the same cluster first:

- Sort all `.cluster.json` by content_hash (stable).
- Each worker shifts its start index by `hash(worker_id) % N`. This is a hint, not a lock — the O_EXCL claim is still the correctness primitive.

### Async within a cluster

Inside `process_cluster`, the per-row pipeline is:

```
row_ids = cluster.row_ids  # e.g. 4 rows for a 4-way duplicate
web_evidences = await asyncio.gather(*[web_search_for(r) for r in row_ids])
bundles = await asyncio.gather(*[ollama_differentiate(r, ev) for r, ev in zip(row_ids, web_evidences)])
```

- `web_search_for`: returns cached hit if `cache/web_search/<sha256(query)>.json` exists; else issues HTTP call.
- `ollama_differentiate`: calls the cloud model for one row's differentiation request.
- Both run concurrently per-row within the cluster. Cluster-level concurrency is bounded by the worker count; row-level concurrency is bounded by a per-worker `asyncio.Semaphore(n)` (default 4).

### Preflight — build the cluster manifest once

Single-shot, before any worker starts:

```
python scripts/ingestion/d3/differentiate_b7_residual.py \
    --build-manifest \
    --violations scripts/ingestion/staging/D3_dedup/re_audit_d3/violations.jsonl \
    --min-cluster 2 --max-cluster 50 \
    --out-dir scripts/ingestion/staging/D3_dedup/differentiate_b7/clusters/
```

Writes one `<content_hash>.cluster.json` per cluster. Idempotent: skips files that already exist. This is the ONLY stage that reads violations.jsonl; workers read the cluster manifests directly.

### Final merge pass — single-threaded, deterministic

After all clusters are done (or a `--stop-after <seconds>` budget elapses):

```
python scripts/ingestion/d3/differentiate_b7_residual.py \
    --merge \
    --merged-in  scripts/ingestion/staging/D3_dedup/merged_stars.jsonl \
    --merged-out scripts/ingestion/staging/D3_dedup/merged_stars.jsonl.next \
    --enriched-dir scripts/ingestion/staging/D3_dedup/differentiate_b7/enriched/ \
    --unresolved-dir scripts/ingestion/staging/D3_dedup/differentiate_b7/unresolved/
```

Steps:
1. Walk `enriched/*.jsonl`, build a `{row_id: enriched_row}` dict.
2. Stream `merged_stars.jsonl` line-by-line; for each row, if the row_id is in the dict, emit the enriched version, else emit the original.
3. Validate: every `done/*.done` file must correspond to either an `enriched/*.jsonl` with resolved rows OR an `unresolved/*.jsonl` (or both for partial resolution). Any mismatch is a hard error.
4. Concat `unresolved/*.jsonl` → `differentiate_b7_unresolved.jsonl` (one consolidated file as the next spec's input).

The merge is deterministic: same inputs → same output bytes. No concurrency here.

---

## Worker lifecycle — no daemon, just CLI

Workers are plain Python processes. Launch as many as cloud rate-limits allow (start with 4, scale up if latency budget permits):

```
python scripts/ingestion/d3/differentiate_b7_residual.py \
    --worker --worker-id worker-$HOSTNAME-$$ \
    --cluster-dir scripts/ingestion/staging/D3_dedup/differentiate_b7/clusters/ \
    --out-root    scripts/ingestion/staging/D3_dedup/differentiate_b7/ \
    --row-concurrency 4 \
    --stop-after 3600
```

- `--worker-id` must be unique per process. Default: `$HOSTNAME-$$-$(date +%s)`.
- `--stop-after`: soft deadline; worker stops claiming new clusters after this many seconds. Releases any held claim and exits 0. Lets operators bound runtime.
- `--row-concurrency`: per-worker asyncio semaphore bound for row-level calls within a cluster. Keep conservative (2-4) until rate-limit behavior is observed.

Multiple workers can run on the same host or across hosts as long as the staging dir is shared filesystem (same mount). NFS is acceptable because O_EXCL over NFSv4 with close-to-open consistency is the claim primitive and it holds.

Operators can launch in background with `nohup ... &`, watch `workers/*.log`, and kill workers individually. Killing a worker mid-cluster leaves a stale claim; the TTL-based steal recovers it within 20 min.

---

## Progress visibility

Each worker's `workers/<worker_id>.log` emits JSONL lines:

```
{"t": "...", "event": "claim",      "cluster": "<hash>", "size": N}
{"t": "...", "event": "web_cache",  "cluster": "<hash>", "hits": K, "misses": K2}
{"t": "...", "event": "bundle",     "cluster": "<hash>", "row": "...", "status": "resolved"|"unresolvable"}
{"t": "...", "event": "done",       "cluster": "<hash>", "resolved": N_ok, "unresolved": N_nope}
{"t": "...", "event": "error",      "cluster": "<hash>", "exc": "..."}
```

A tiny aggregator command for Daniel:

```
python scripts/ingestion/d3/differentiate_b7_residual.py --status
```

Reads `done/*.done` + `claims/*.claim` + cluster count, prints:

```
clusters_total=3083 done=N claims_live=M unclaimed=3083-N-M
rows_resolved=X rows_unresolved=Y
eta_seconds=... (based on rolling throughput from worker logs)
```

---

## What NOT to change in this commit

- **Do NOT** touch the proceduralizer contract, `mcp_web_search.py`, or the differentiation prompt. Steps 1-2 are green; don't reopen them.
- **Do NOT** couple the parallel driver to any job queue, Redis, or external broker. The shared filesystem IS the queue.
- **Do NOT** merge the contaminated commit `2ded4dc3` into this work. Address it separately (see §"Commit hygiene" below).
- **Do NOT** add a TTL shorter than 20 min. Cloud calls can legitimately take 5-8 min under load; shorter TTLs cause claim-steal races.
- **Do NOT** auto-rerun `bash scripts/ingestion/d3/run.sh` at the end of the merge pass. That's a separate verification step the operator triggers.

---

## Sovereignty note

Everything here is ingestion-path filesystem + asyncio + cloud HTTP. Allowed per CLAUDE.md "Ingestion Path = Flexible". The hot path is not touched, the kernel is not touched, the registry is not touched. Output artifact (merged_stars.jsonl) remains the sole interface to the sovereign path.

---

## Execution order

### Step 1 — Commit hygiene (resolve the contamination BEFORE the parallel work)

**The problem**: commit `2ded4dc3` is contaminated with unrelated staged worktree changes. The pre-commit hook was also bypassed because `.git/hooks/pre-commit` is not executable in this clone.

**Fix the hook first** (no history rewrite):

```bash
# If a tracked pre-commit source exists, re-link it:
ls -la .git/hooks/pre-commit scripts/git-hooks/ 2>&1
# If scripts/git-hooks/pre-commit exists:
ln -sf ../../scripts/git-hooks/pre-commit .git/hooks/pre-commit
chmod +x .git/hooks/pre-commit
# If no tracked source, just make the existing one executable:
chmod +x .git/hooks/pre-commit
# Verify: the hook now runs on the NEXT commit
```

If Daniel has a tracked hook source, use the symlink form. Otherwise the chmod is sufficient for this clone. Do NOT commit anything in `.git/` — that's local-only.

**Split the contaminated commit** — pick ONE of these based on whether `2ded4dc3` has been pushed:

- **Option A — not pushed yet (preferred)**: rewrite the two B7 commits as a clean series.
  ```bash
  git log --oneline main..HEAD   # confirm 2ded4dc3 and cc83d67c are local
  git reset --soft 2ded4dc3~1    # un-commit both, keep changes staged
  git reset HEAD                 # unstage everything
  # Now stage ONLY the B7 contract + web_search files for commit #1:
  git add knowledge3d/ingestion/proceduralizer_contract.py \
          knowledge3d/ingestion/proceduralizer_wine.py \
          knowledge3d/ingestion/mcp_web_search.py \
          knowledge3d/tools/knowledge_proceduralizer.py \
          knowledge3d/knowledgeverse/proceduralizer_stargate.py \
          docs/vocabulary/KNOWLEDGE_PROCEDURALIZER_SPECIFICATION.md
  git diff --cached --stat       # verify: no unrelated files
  git commit -m "B7 add cloud differentiation contract and web search"
  # Stage ONLY the driver for commit #2:
  git add scripts/ingestion/d3/differentiate_b7_residual.py
  git diff --cached --stat       # verify: driver only
  git commit -m "B7 add residual differentiation driver"
  # Now any contaminating unrelated changes are back in the worktree
  # untracked; Daniel reviews and stages them in their own commits.
  git status
  ```
- **Option B — already pushed**: do NOT force-push. Leave `2ded4dc3` in history, write a follow-up commit `git revert <unrelated-hashes>` style or a `git restore` commit that removes the unrelated additions from the tree. Document the contamination in the next commit message.

Confirm with Daniel which branch state applies before rewriting. The greps that tell you:
```bash
git log --oneline origin/<this-branch> 2>/dev/null | grep 2ded4dc3 && echo "pushed" || echo "local only"
```

### Step 2 — Land the parallel driver

Extend `scripts/ingestion/d3/differentiate_b7_residual.py` with the four new modes:
- `--build-manifest` — preflight cluster manifest builder
- `--worker` — claim-loop worker (async per-row inside cluster)
- `--merge` — final single-threaded merge pass
- `--status` — aggregator for Daniel

Add a small `scripts/ingestion/d3/launch_b7_workers.sh` that launches N workers in the background with unique worker_ids and tails the aggregate status. N defaults to 4.

Commit message template:

```
B7 parallel differentiation driver (claim-based, no re-work, no overlap)

Adds --build-manifest, --worker, --merge, --status modes to
differentiate_b7_residual.py. Workers claim clusters via O_EXCL lock files
and write per-cluster outputs so a final single-threaded merge is
deterministic. Row-level calls inside a cluster run concurrently under a
per-worker asyncio.Semaphore.

Guarantees:
- No overlap: O_EXCL on claims/<hash>.claim is the serialization primitive.
- No re-work: done/<hash>.done is checked before every cluster and before
  every claim retry. Completed clusters are skipped globally, including
  across separate runs.
- Resumable: killing a worker leaves a stale claim that another worker
  steals after a 20-min TTL.

Spec: TEMP/CLAUDE_CODEX_B7_PARALLEL_DIFFERENTIATION_DRIVER_04.19.2026.md
```

### Step 3 — Build the manifest (one-shot, minutes)

```bash
python scripts/ingestion/d3/differentiate_b7_residual.py \
  --build-manifest \
  --violations scripts/ingestion/staging/D3_dedup/re_audit_d3/violations.jsonl \
  --min-cluster 2 --max-cluster 50 \
  --out-dir scripts/ingestion/staging/D3_dedup/differentiate_b7/clusters/
```

Expected: 3,083 cluster files written. Report the count in the terminal.

### Step 4 — Launch workers (background, hours)

```bash
bash scripts/ingestion/d3/launch_b7_workers.sh 4 3600
# 4 workers, 1-hour budget per worker. Adjust if cloud rate-limits allow.
```

Workers emit per-worker logs in `workers/`. Monitor with:

```bash
watch -n 30 'python scripts/ingestion/d3/differentiate_b7_residual.py --status'
```

When `done/` count == cluster count (or the budget elapses), stop and move on. Do not extend runtime past the observed ETA × 1.5.

### Step 5 — Merge (one-shot, minutes)

```bash
python scripts/ingestion/d3/differentiate_b7_residual.py \
  --merge \
  --merged-in  scripts/ingestion/staging/D3_dedup/merged_stars.jsonl \
  --merged-out scripts/ingestion/staging/D3_dedup/merged_stars.jsonl.next \
  --enriched-dir scripts/ingestion/staging/D3_dedup/differentiate_b7/enriched/ \
  --unresolved-dir scripts/ingestion/staging/D3_dedup/differentiate_b7/unresolved/
```

Then swap and re-audit:

```bash
mv scripts/ingestion/staging/D3_dedup/merged_stars.jsonl{.next,}
bash scripts/ingestion/d3/run.sh --recover-only
```

Report to me:
- `clusters_attempted / clusters_resolved`
- `rows_enriched / rows_unresolved`
- new `duplicate_row_count`
- new `merged_stars.jsonl` sha256
- cloud wall-clock time + web_cache hit rate

### Step 6 — Cross-check the new .bin (after Step 5 lands)

```bash
bash scripts/ingestion/d3/run.sh
sha256sum scripts/ingestion/staging/D3_dedup/matryoshka_embeddings.bin
scripts/ingestion/d3/build/matryoshka_bin_producer \
  --input  scripts/ingestion/staging/D3_dedup/merged_stars.jsonl \
  --output /tmp/cc_cpp.bin \
  --force-regenerate --verbose
sha256sum /tmp/cc_cpp.bin
# Both hashes must match byte-for-byte.
```

---

## Failure modes + mitigations

| Failure | Mitigation |
|---|---|
| Worker crashes mid-cluster | 20-min claim TTL; next worker steals the claim. No row is processed twice because claim-steal + `done/` check are serialized. |
| Cloud rate-limit 429 | Per-worker asyncio semaphore; back off with exponential jitter. Log as `event=rate_limit` and pause that worker only. |
| Web search returns empty | Bundle emits `status=unresolvable`. Row is logged to `unresolved/`. No fabrication. |
| Same content_hash appears in two clusters (shouldn't happen but defensively) | Manifest builder deduplicates by content_hash on write. |
| Worker writes partial `enriched/<hash>.jsonl` then crashes | `done/<hash>.done` is written AFTER `enriched/<hash>.jsonl` is fsynced and renamed. No `done` → claim is stealable → cluster re-runs. Partial enriched file is overwritten on re-run. |
| Operator kills all workers | Re-run `launch_b7_workers.sh`. Unclaimed + stale-claimed clusters are picked up; completed ones are skipped via `done/` check. |

---

## Related files

- `scripts/ingestion/d3/differentiate_b7_residual.py` — extend with four new modes
- `scripts/ingestion/d3/launch_b7_workers.sh` — NEW bash launcher
- `scripts/ingestion/staging/D3_dedup/differentiate_b7/` — NEW staging tree
- `scripts/ingestion/staging/D3_dedup/re_audit_d3/violations.jsonl` — input (read-only)
- `knowledge3d/ingestion/mcp_web_search.py` — unchanged
- `knowledge3d/tools/knowledge_proceduralizer.py` — unchanged (differentiate_cluster function already exists)

---

**Estimated effort**: Step 1 ~30min (hygiene), Step 2 ~3-4h (driver), Steps 3-6 ~run-time + ~1h active verify.
**Blocks**: Gate 1 full close-out and the "sovereign procedural symlinked architecture live end-to-end" milestone.
**Blocked by**: Nothing. Steps 1-2 of the B7 cloud proceduralizer spec are landed.
**Location**: `TEMP/CLAUDE_CODEX_B7_PARALLEL_DIFFERENTIATION_DRIVER_04.19.2026.md`
