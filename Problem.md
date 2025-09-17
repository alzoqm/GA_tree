# Overfitting Investigation Notes

## Symptoms
- Models chosen by WFO show strong in-sample metrics yet collapse on full out-of-sample evaluation; best checkpoints repeat a single window’s behavior.
- Aggregated OOS metrics such as consistency/DSR are almost binary, signaling that per-window history is not accumulated correctly.

## Root Cause
- `training/wfo.py:577-580` creates per-window IDs (`"w{window_idx}_c{candidate_idx}"`), so `record_window_result` treats each window independently. 
- `aggregate_candidate_objectives` (`training/wfo.py:190-258`) expects a consistent candidate ID across windows, but never receives it; final ranking therefore reflects only one window.
- Structural regularization is effectively off: `is_objectives` omits complexity, and pruning thresholds (`experiment_config.yaml:274-293`) are set so high that pruning never fires. Trees grow large, further hurting generalization.

## Planned Fixes
- Introduce deterministic candidate IDs (e.g., hash the chromosome tensor) before calling `record_window_result` so identical individuals accumulate all-window results.
- Re-evaluate top candidates from prior windows when advancing WFO to ensure every survivor has a complete OOS history.
- Reinstate complexity pressure (enable complexity objective, tune pruning thresholds) and adjust OOS consistency thresholds (`experiment_config.yaml:256-270`) to penalize inconsistent performers.
- Add a final holdout evaluation before saving top models to catch residual overfitting.
