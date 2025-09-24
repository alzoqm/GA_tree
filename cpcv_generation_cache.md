# CPCV Generation Cache Report

## 1. Diagnostic Summary
- **Redundant GPU simulations:** The current CPCV evaluator runs the simulator once per `(train, test)` slice for every combination, so a 10-fold, 2-test setup triggers roughly 220 GPU launches per generation even though only 34 unique contiguous slices exist.
- **Transient insight loss:** Because each simulation is discarded after a single combination, there is no archival trail for downstream diagnostics or reproducibility checks.
- **In-memory bias:** Ad-hoc segment caching inside the evaluator keeps duplicate `DataFrame` slices but still replays the full GPU workload, leaving the primary bottleneck untouched.

## 2. Why This Matters
- **Runtime blow-up:** The quadratic growth of `C(folds, test_folds)` quickly saturates GPU throughput and drags generation time, affecting the ability to iterate on model design.
- **Resource contention:** Repeated CUDA launches increase contention for device memory and kernel scheduling; schedulers with `parallel_chunks > 1` amplify the problem.
- **Checkpoint fragility:** With no reusable artifacts, a mid-generation interruption forces every split to be recomputed, undermining the checkpointing groundwork already present for CPCV mode.

## 3. Proposed Generation-Level Cache
Use a deterministic cache that stores simulation outputs per unique segment once per generation and reassembles combination metrics from those cached payloads.

### Example Scenario
- Population size: 512 strategies.
- Dataset length: 120,000 bars split into 10 chronological folds.
- CPCV setup: 2 test folds with 600-bar purge and embargo.
- **Discovery:** Walking the scheduler reveals 10 unique test slices (one per fold) and 24 unique train slices (purge/embargo variants). The cache therefore caps GPU simulations at 34 per generation rather than ~220.

### Segment Key Definition
- `SegmentKey = (kind, start_idx, stop_idx, purge_before, embargo_after, population_hash, feature_hash)` ensures that cached results correspond to the exact data window and population snapshot.
- The `population_hash` can be produced once per generation using a deterministic tensor digest (e.g., hashing `population.population_tensor.cpu().numpy().tobytes()` with `hashlib`).
- `feature_hash` covers feature order or any stochastic preprocessing choices so that the cache is invalidated automatically when feature pipelines change.

## 4. Implementation Blueprint

### Stage 1 · Enumerate Unique Segments
- Traverse `CPCVScheduler` once at the start of `CPCVObjectiveAggregator.evaluate` and map every train/test slice it emits.
- Deduplicate contiguous ranges using the `SegmentKey`. Keep reference counters listing which combinations and in what chronological order consume each key.
- Store this metadata alongside the generation checkpoint so that a resume operation can rebuild the cache lazily.

### Stage 2 · Populate the Cache
- For each unique key, run the simulator **once** for the full population with returns, benchmark series, and node statistics enabled.
- Materialize outputs under `evaluation_cache_dir/generation_<id>/segments/segment_<hash>.{npz|mm}`. Use `np.memmap` for long sequences to avoid loading the entire matrix into memory at once.
- Persist metadata (length, dtype, reference count) in a lightweight manifest JSON so the evaluator can reopen segments on demand.

### Stage 3 · Assemble Combination Metrics
- When evaluating a combination, fetch the ordered list of segment keys, read their cached payloads (memory-map if needed), and concatenate returns chronologically.
- Recompute path-dependent indicators (e.g., max drawdown, Calmar ratio) from the merged return stream. `_accumulate_env_state` should only aggregate additively safe quantities such as sums and counts.
- Apply existing objective reducers (`median`, `IQR`, `CVaR`, turnover) exactly as today; only the data source changes from live simulation to cached tensors.

### Stage 4 · Evict and Finalize
- Decrement each segment’s reference count after its last consuming combination completes. Immediately release memory-mapped handles or delete in-memory arrays once the count reaches zero.
- At generation end, move the manifest and any remaining cached artifacts into a `completed` subdirectory to support audit trails, then clear the active cache before the next generation begins.

## 5. Potential Issues and Mitigations
- **Memory management:** Even with memmap, short-lived hot segments may sit in RAM. Maintain a configurable ceiling (e.g., total cached bytes) and evict the longest-idle segments first. Reference counting ensures segments disappear as soon as their final consumer finishes.
- **Path-dependent metrics:** Never rely on summed `env_state` fields for metrics that depend on sequence order. Always rebuild the cumulative equity curve from concatenated returns before computing drawdowns, Calmar, or Omega ratios.
- **Cache key integrity:** Include population and feature hashes in the key so cached results cannot leak across generations or mutated populations. Any mutation toggles the hash and forces a fresh simulation.
- **Disk throughput:** Writing dozens of large return matrices can become I/O-bound on spinning disks. Batch writes where possible and prefer chunked binary formats (npz with compression disabled) to minimize CPU overhead.
- **Parallel access:** When `parallel_chunks` spawns concurrent workers, guard the manifest updates with lightweight file locks to avoid race conditions while decrementing reference counts.
- **Failure recovery:** On crash restart, replay the manifest to reload any segments already persisted, saving time on partial generations.

## 6. Action Items
- Instrument the existing evaluator to export segment manifests and hashes before simulation begins.
- Build a `GenerationSimulationCache` utility responsible for deduplication, reference tracking, persistence under `evaluation_cache_dir`, and cleanup.
- Update CPCV checkpoints to include cache manifests so interrupted runs resume without recomputing completed segments.
- Add profiling hooks to compare GPU launch counts and runtime before and after introducing the cache.

With these changes, CPCV training reuses simulations across all individuals in a generation, curbs redundant GPU workloads, and stores reusable artifacts under `evaluation_cache_dir` without compromising metric correctness.