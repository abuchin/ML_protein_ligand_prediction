# Plan: Optimize, Debug & Harden ML Protein-Ligand Prediction Pipeline

## Context
A three-agent audit identified 22 issues across the codebase: correctness bugs (including a data-leakage bug in vocabulary construction), broken parallelization (ProcessPoolExecutor imported but never called), config parameters defined but silently ignored, dead SHAP code, and zero test coverage on core modules. This plan fixes all four categories top-down by severity.

---

## Phase 1 — Bugs & Correctness (Highest Priority)

### 1.1 Data leakage: auxiliary feature vocabulary built on all proteins
**File:** `src/plbind/data/protein_fetcher.py` ~line 118–121  
**Issue:** GO-term and Pfam vocabularies are built over all proteins passed to `build_auxiliary_features()` — including test proteins. Downstream one-hot encodings of test proteins can influence training-set encodings.  
**Fix:** Accept an optional `fit_vocab=True` parameter. When `False`, skip `fit()` and only `transform()` with the existing vocabulary. In `TrainingPipeline`, call with `fit_vocab=True` on train proteins, then re-call with `fit_vocab=False` on val/test proteins.

### 1.2 Config `protein_max_length` silently ignored
**File:** `scripts/run_data_prep.py` ~line 161–165  
**Issue:** `CFG.protein_max_length = 1022` is defined in `config.py` but `ESM2Encoder` is instantiated with the hardcoded default.  
**Fix:** Pass `max_length=CFG.protein_max_length` explicitly at instantiation.

### 1.3 Config parameters not propagated — `test_size`, `val_size`, `cv_folds`
**Files:** `src/plbind/training/pipeline.py` ~lines 97, 172  
**Issue:** `Splitter` is instantiated without `test_size`/`val_size` from CFG; CV uses hardcoded `cv=5` instead of `CFG.cv_folds`.  
**Fix:**
- `Splitter(test_size=CFG.test_size, val_size=CFG.val_size, random_state=self.random_seed)`
- Replace `cv=5` → `cv=CFG.cv_folds` in the `cross_validate()` call.

### 1.4 Unused config flags (`morgan_use_counts`, `use_maccs`, `use_atompair`)
**Files:** `src/plbind/config.py`, `src/plbind/data/ligand_encoder.py`  
**Issue:** Boolean flags in CFG are defined but encoder always runs all four fingerprint types unconditionally.  
**Fix:** In `LigandEncoder._encode_one()`, guard each fingerprint block behind its CFG flag. Update `total_fp_bits` calculation accordingly.

### 1.5 Dead code — unused `.ToList()` call
**File:** `src/plbind/data/ligand_encoder.py` ~line 213  
**Issue:** `fp.ToList()` return value is discarded; `ConvertToNumpyArray()` is what extracts the array.  
**Fix:** Remove the `.ToList()` call entirely.

### 1.6 MLP early stopping patience mismatch
**Files:** `src/plbind/models/mlp.py` (default `patience=10`), `src/plbind/training/pipeline.py` (passes `CFG.patience=15`)  
**Issue:** Direct use of `InteractionMLP` outside the pipeline gets a different default than the pipeline path.  
**Fix:** Change `mlp.py` default `patience` to match `CFG.patience` (15), or read `CFG.patience` directly.

### 1.7 LightGBM callbacks robustness
**File:** `src/plbind/models/lightgbm_model.py` ~line 77–82  
**Issue:** On `ImportError` the callbacks list is `[]`, but `callbacks or None` evaluates to `None` correctly — however, this branch is not logged; silent fallback.  
**Fix:** Add an explicit `logger.warning("early stopping callbacks unavailable...")` in the except block so failures are visible.

---

## Phase 2 — Performance & Optimizations

### 2.1 Fix broken parallelization in ligand encoder (highest-impact)
**File:** `src/plbind/data/ligand_encoder.py` ~lines 148–162  
**Issue:** `encode_batch()` builds chunks and imports `ProcessPoolExecutor` but runs a sequential list comprehension instead of submitting to the executor. Expected 4–8× speedup on large datasets.  
**Fix:**
```python
with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
    results = list(executor.map(_encode_chunk, chunks))
```
Wrap in try/except to fall back to sequential if multiprocessing fails (Windows/macOS fork restrictions).

### 2.2 Keep sparse matrices sparse through sklearn pipeline
**File:** `src/plbind/features/feature_builder.py` ~line 128  
**Issue:** `ligand_fp_block.toarray()` converts CSR sparse (2214 dims) to dense on every `build()` call, negating the sparse storage benefit.  
**Fix:** Use `scipy.sparse.hstack` to concatenate the dense protein/aux blocks (converted via `np.hstack` or `scipy.sparse.csr_matrix(dense_part)`) with the sparse fingerprint block. LightGBM and XGBoost handle sparse matrices natively. For LR and RF, convert to dense only at model fit time inside `_train_model()`. Expose a `return_sparse: bool` parameter on `build()`.

### 2.3 Replace `iterrows()` in feature builder
**File:** `src/plbind/features/feature_builder.py` ~line 167  
**Issue:** Iterating `df.iterrows()` row-by-row is 10–100× slower than vectorized `.loc[]` indexing.  
**Fix:** Replace the loop with a vectorized lookup using `df.index` and `df[col].values` or `df.loc[indices, col]`.

### 2.4 Add `ReduceLROnPlateau` patience to config
**File:** `src/plbind/models/mlp.py` ~line 260  
**Issue:** `patience=3` for the LR scheduler is hardcoded.  
**Fix:** Add `lr_scheduler_patience: int = 3` to `CFG` and use it here.

---

## Phase 3 — SHAP Integration

### 3.1 Wire SHAPAnalyzer into TrainingPipeline
**Files:** `src/plbind/training/pipeline.py`, `src/plbind/evaluation/interpretability.py`  
**Issue:** `SHAPAnalyzer` is fully implemented (TreeExplainer, DeepExplainer, feature group aggregation) but never invoked anywhere.  
**Fix:**
1. After each sklearn model is trained in the pipeline, instantiate `SHAPAnalyzer(model.model, X_train_sample)` where `X_train_sample = X_train[:CFG.shap_background_samples]`.
2. Call `shap_vals = analyzer.explain_tree(X_test)` (or `explain_mlp()` for MLP).
3. Save SHAP values to `outputs/shap/{model_name}_shap_values.npy`.
4. Call `analyzer.feature_group_importance(shap_vals, block_map)` and add results to `results.json` under `"shap"` key.
5. Call `analyzer.plot_summary(shap_vals, feature_names, save_path="outputs/figures/shap_{model_name}.png")` — change `plt.show()` to use `save_path` kwarg so it works headlessly.
6. Guard entire block with `try/except ImportError` so SHAP remains optional.

### 3.2 Fix `plt.show()` unconditional calls
**File:** `src/plbind/evaluation/interpretability.py` ~lines 157, 175  
**Fix:** Add `show: bool = False` parameter; only call `plt.show()` when `show=True`. Always save to file when `save_path` is provided.

---

## Phase 4 — Tests

### 4.1 Tests for Splitter strategies
**New file:** `tests/test_splitter.py`  
Cover: `random_split`, `cold_protein_split`, `cold_ligand_split`, `cold_both_split`.  
Key assertions:
- No protein ID overlap between train and test in cold-protein splits.
- No ligand CID overlap between train and test in cold-ligand splits.
- Positive rate preserved (stratification check: abs diff < 5%).

### 4.2 Tests for DataPreprocessor
**New file:** `tests/test_preprocessor.py`  
Cover: label creation at threshold boundary, decoy generation (count ratio, no CID collision with binders), `keep_only_measured` filtering.

### 4.3 Tests for feature_builder block_map correctness
**New test in:** `tests/test_feature_engineer.py`  
Cover: `block_map` slices sum to total feature dimensions, each block is non-overlapping, protein slice has expected width for given pooling strategy.

### 4.4 Tests for SHAPAnalyzer (post integration)
**New file:** `tests/test_interpretability.py`  
Cover: `explain_tree()` returns array of shape `(n_samples, n_features)`, `feature_group_importance()` returns dict with expected group keys, `plot_summary()` saves file without `plt.show()`.

### 4.5 Upgrade test fixtures to reflect real data shapes
**File:** `tests/conftest.py`  
Change synthetic fixture from `(200, 20)` to a shape that mirrors the actual feature layout:
- `n_samples=500`, `n_features=4884` (or parameterized).
- Add a sparse fixture for fingerprint-only matrices.

---

## Critical Files

| File | Changes |
|------|---------|
| `src/plbind/data/protein_fetcher.py` | fit_vocab parameter for vocab leakage fix |
| `src/plbind/data/ligand_encoder.py` | Fix ProcessPoolExecutor, remove dead ToList(), honor CFG flags |
| `src/plbind/features/feature_builder.py` | Keep sparse, fix iterrows, expose return_sparse |
| `src/plbind/training/pipeline.py` | Propagate CFG.test_size/val_size/cv_folds, wire SHAP |
| `src/plbind/models/mlp.py` | Fix patience default, add lr_scheduler_patience config |
| `src/plbind/models/lightgbm_model.py` | Add logger.warning on callback fallback |
| `src/plbind/evaluation/interpretability.py` | Fix plt.show(), add save_path kwarg |
| `scripts/run_data_prep.py` | Pass CFG.protein_max_length to ESM2Encoder |
| `src/plbind/config.py` | Add lr_scheduler_patience field |
| `tests/test_splitter.py` | New file |
| `tests/test_preprocessor.py` | New file |
| `tests/test_interpretability.py` | New file |
| `tests/conftest.py` | Update fixture shapes |

---

## Execution Order

1. **Phase 1** (bugs) — do all fixes before running any benchmarks
2. **Phase 2** (performance) — 2.1 (parallelization) first as it's highest-impact; 2.2 (sparse) second
3. **Phase 3** (SHAP) — after pipeline is stable
4. **Phase 4** (tests) — write tests in parallel with Phase 2/3, run at end to verify all fixes

## Verification

```bash
# After all changes:
pytest tests/ -v                          # all tests green
python scripts/run_data_prep.py --n_samples 200   # data prep runs cleanly
python scripts/run_training.py --n_samples 200 --split cold_protein  # quick smoke test

# Final benchmark: 1000-protein cold_both split
python scripts/run_training.py --n_proteins 1000 --split cold_both
# Check outputs/results.json has "shap" key
# Check outputs/figures/ has shap_*.png files
```
