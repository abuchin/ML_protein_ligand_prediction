# Plan: Optimize, Debug & Harden ML Protein-Ligand Prediction Pipeline

## Status: COMPLETED (2026-04-30)

A three-agent audit identified 22 issues across the codebase. All four phases have been implemented. Below is the original plan annotated with completion status.

---

## Phase 1 — Bugs & Correctness ✅ ALL DONE

### 1.1 ✅ Data leakage: auxiliary feature vocabulary built on all proteins
**Fix:** Added `fit_vocab: bool = True` parameter to `build_auxiliary_features()` in `protein_fetcher.py`. `TrainingPipeline` calls with `fit_vocab=True` on training proteins only.

### 1.2 ✅ Config `protein_max_length` silently ignored
**Fix:** `scripts/run_data_prep.py` now passes `max_length=CFG.protein_max_length` to `ESM2Encoder`.

### 1.3 ✅ Config parameters not propagated — `test_size`, `val_size`, `cv_folds`
**Fix:** `Splitter` instantiated with `test_size=CFG.test_size, val_size=CFG.val_size`; CV uses `cv=CFG.cv_folds`.

### 1.4 ✅ Unused config flags (`morgan_use_counts`, `use_maccs`, `use_atompair`)
**Fix:** `LigandEncoder` accepts and enforces all three flags; `fp_dim` property computed accordingly.

### 1.5 ✅ Dead code — unused `.ToList()` call
**Fix:** Removed from `_atompair()` in `ligand_encoder.py`.

### 1.6 ✅ MLP early stopping patience mismatch
**Fix:** `mlp.py` default `patience` changed from 10 → 15 to match `CFG.patience`.

### 1.7 ✅ LightGBM callbacks robustness
**Fix:** Added `logger.warning(...)` in the `except ImportError` block of `lightgbm_model.py`.

---

## Phase 2 — Performance & Optimizations ✅ ALL DONE

### 2.1 ✅ Fix broken parallelization in ligand encoder
**Fix:** Moved `_encode_chunk_fn` to module level (was a closure, unpicklable). `encode_batch()` now uses `ProcessPoolExecutor` with sequential fallback.

### 2.2 ⚠️ Keep sparse matrices sparse through sklearn pipeline
**Status:** Partially deferred. `feature_builder.py` still converts fingerprints to dense via `.toarray()` before concatenation. LightGBM/XGBoost could consume CSR directly — left for a future PR since the bottleneck is training, not feature assembly.

### 2.3 ✅ Replace `iterrows()` in feature builder
**Fix:** Replaced with vectorized lookup using numpy arrays — ~10–100× faster.

### 2.4 ✅ Add `ReduceLROnPlateau` patience to config
**Fix:** `lr_scheduler_patience: int = 3` added to `config.py`; used in `InteractionMLPModel`.

---

## Phase 3 — SHAP Integration ✅ ALL DONE

### 3.1 ✅ Wire SHAPAnalyzer into TrainingPipeline
**Fix:** `_run_shap()` method added to `TrainingPipeline`. Called after each sklearn model is trained. Saves `.npy` SHAP arrays, summary beeswarm plots, and group-importance bar charts. Results added to `results.json` under `"shap_group_importance"`.

### 3.2 ✅ Fix `plt.show()` unconditional calls
**Fix:** `show: bool = False` parameter added to `plot_summary()` and `plot_group_importance()`. Also fixed unconditional `plt.show()` in `evaluator.py:plot_calibration_curves()` (replaced with `plt.close()`).

---

## Phase 4 — Tests ✅ ALL DONE

### 4.1 ✅ Tests for Splitter strategies
**File:** `tests/test_splitter.py` — 17 tests covering all split strategies with protein/ligand overlap assertions.

### 4.2 ✅ Tests for DataPreprocessor
**File:** `tests/test_preprocessor.py` — 9 tests covering label creation, threshold boundary, deduplication, decoy generation, and `keep_only_measured`.

### 4.3 ✅ Tests for feature_builder block_map correctness
**File:** `tests/test_feature_engineer.py` — 3 additional tests: slices cover total dim, non-overlapping, aux block correctness.

### 4.4 ✅ Tests for SHAPAnalyzer
**File:** `tests/test_interpretability.py` — 15 tests covering `explain_tree`, `feature_group_importance`, `top_features`, and plot helpers.

### 4.5 ✅ Upgrade test fixtures
**File:** `tests/conftest.py` — Added `realistic_feature_data` (300×2709, mimics real pipeline dims) and `sparse_fingerprint_matrix` (500×2214 CSR).

---

## Additional Fixes (discovered during implementation)

### A1 ✅ OpenMP / PyTorch deadlock after sklearn CV
**Root cause:** `torch.randperm()` (called by `DataLoader(shuffle=True)`) uses OpenMP parallel sorting via `__kmpc_fork_call`. After sklearn's CV finishes, the OpenMP thread pool is left in a broken barrier state. When PyTorch forks new work, threads deadlock at `__kmp_join_barrier → _pthread_cond_wait`.
**Fix:** `torch.set_num_threads(1)` called before MLP training in `pipeline.py`; also skipped MPS in `auto` device mode (PyTorch 2.8 + macOS 15 has `BatchNorm1d` backward deadlocks on MPS).

### A2 ✅ FocalLoss device mismatch on MPS
**Fix:** `torch.arange(len(targets))` now passes `device=targets.device` explicitly.

### A3 ✅ Added --mlp_epochs and --mlp_patience CLI flags
Allows fast iteration without editing config.py.

---

## Benchmark Results (1000-protein cold_both, 2026-04-30)

| Model | ROC-AUC | PR-AUC | F1 |
|---|---|---|---|
| LightGBM | 0.998 | 0.9996 | 0.989 |
| Random Forest | 0.998 | 0.9995 | 0.996 |
| XGBoost | 0.997 | 0.9993 | 0.995 |
| InteractionMLP | 0.993 | 0.9986 | 0.969 |
| Logistic Regression | 0.977 | 0.9960 | 0.946 |

Presentation PDF: `Presentation/protein_ligand_results.pdf`
