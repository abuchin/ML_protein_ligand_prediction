"""Tests for DataPreprocessor: labels, negatives, duplicate handling."""
import numpy as np
import pandas as pd
import pytest

from plbind.data.preprocessor import DataPreprocessor


def _write_csv(tmp_path, df: pd.DataFrame, name: str = "data.csv"):
    path = tmp_path / name
    df.to_csv(path, index=False)
    return str(path)


def _synthetic_raw(n: int = 100, seed: int = 42) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    return pd.DataFrame({
        "UniProt_ID": [f"P{i % 10:04d}" for i in range(n)],
        "pubchem_cid": rng.randint(1, 30, n),
        "kiba_score": rng.uniform(0, 30, n),
        "kiba_score_estimated": rng.choice([True, False], n),
    })


class TestLabelCreation:
    def test_positive_below_threshold(self, tmp_path):
        df = _synthetic_raw(200)
        path = _write_csv(tmp_path, df)
        prep = DataPreprocessor(kiba_threshold=12.1, kiba_binder_is_below=True,
                                use_property_matched_decoys=False, negative_ratio=0.0)
        result = prep.preprocess(path)
        real_pos = result[(result["bound"] == 1) & result["kiba_score"].notna()]
        assert (real_pos["kiba_score"] < 12.1).all()

    def test_positive_above_threshold_when_inverted(self, tmp_path):
        df = _synthetic_raw(100)
        path = _write_csv(tmp_path, df)
        prep = DataPreprocessor(kiba_threshold=12.1, kiba_binder_is_below=False,
                                use_property_matched_decoys=False, negative_ratio=0.0)
        result = prep.preprocess(path)
        real_pos = result[(result["bound"] == 1) & result["kiba_score"].notna()]
        assert (real_pos["kiba_score"] > 12.1).all()

    def test_exact_threshold_boundary(self, tmp_path):
        """Only rows below threshold are kept as positives (strict less-than).

        The preprocessor keeps only positives + generated negatives in the output;
        raw negative rows are discarded and replaced by synthetic decoys.
        kiba=12.0 < 12.1 → bound=1 (included); kiba=12.1 and 12.5 → not bound (excluded
        from output when negative_ratio=0.0 means no decoys are generated).
        """
        df = pd.DataFrame({
            "UniProt_ID": ["P0001", "P0002", "P0003"],
            "pubchem_cid": [1, 2, 3],
            "kiba_score": [12.0, 12.1, 12.5],
            "kiba_score_estimated": [False, False, False],
        })
        path = _write_csv(tmp_path, df)
        prep = DataPreprocessor(kiba_threshold=12.1, kiba_binder_is_below=True,
                                use_property_matched_decoys=False, negative_ratio=0.0)
        result = prep.preprocess(path)
        # Only cid=1 (kiba=12.0 < 12.1) should appear as a positive
        assert (result["bound"] == 1).sum() == 1
        assert result.loc[result["bound"] == 1, "pubchem_cid"].iloc[0] == 1

    def test_positive_rate_within_bounds(self, tmp_path):
        df = _synthetic_raw(200)
        path = _write_csv(tmp_path, df)
        prep = DataPreprocessor(kiba_threshold=12.1, use_property_matched_decoys=False,
                                negative_ratio=1.0)
        result = prep.preprocess(path)
        pos_rate = result["bound"].mean()
        assert 0.01 < pos_rate < 0.99


class TestDuplicateHandling:
    def test_duplicates_averaged(self, tmp_path):
        df = pd.DataFrame({
            "UniProt_ID": ["P0001", "P0001", "P0002"],
            "pubchem_cid": [1, 1, 2],
            "kiba_score": [10.0, 14.0, 5.0],  # pair (P0001, 1) averages to 12.0
            "kiba_score_estimated": [False, False, False],
        })
        path = _write_csv(tmp_path, df)
        prep = DataPreprocessor(kiba_threshold=12.1, use_property_matched_decoys=False,
                                negative_ratio=0.0)
        result = prep.preprocess(path)
        row = result[(result["UniProt_ID"] == "P0001") & (result["pubchem_cid"] == 1)]
        assert len(row) == 1
        assert abs(row.iloc[0]["kiba_score"] - 12.0) < 0.01

    def test_unique_pairs_after_dedup(self, tmp_path):
        df = _synthetic_raw(100)
        # add explicit duplicate
        df = pd.concat([df, df.iloc[:10]], ignore_index=True)
        path = _write_csv(tmp_path, df)
        prep = DataPreprocessor(use_property_matched_decoys=False, negative_ratio=0.0)
        result = prep.preprocess(path)
        real = result[result["kiba_score"].notna()]
        dupes = real.duplicated(subset=["UniProt_ID", "pubchem_cid"])
        assert not dupes.any()


class TestNegativeGeneration:
    def test_negative_ratio_approximately_one(self, tmp_path):
        df = _synthetic_raw(200)
        path = _write_csv(tmp_path, df)
        prep = DataPreprocessor(negative_ratio=1.0, use_property_matched_decoys=False,
                                random_state=42)
        result = prep.preprocess(path)
        n_pos = (result["bound"] == 1).sum()
        n_neg = (result["bound"] == 0).sum()
        ratio = n_neg / max(n_pos, 1)
        assert 0.5 <= ratio <= 2.0

    def test_negatives_do_not_collide_with_known_positives(self, tmp_path):
        df = _synthetic_raw(100)
        path = _write_csv(tmp_path, df)
        prep = DataPreprocessor(negative_ratio=1.0, use_property_matched_decoys=False,
                                random_state=0)
        result = prep.preprocess(path)
        positives = result[result["bound"] == 1]
        negatives = result[result["bound"] == 0]
        pos_pairs = set(zip(positives["UniProt_ID"], positives["pubchem_cid"]))
        neg_pairs = set(zip(negatives["UniProt_ID"], negatives["pubchem_cid"]))
        assert len(pos_pairs & neg_pairs) == 0

    def test_zero_ratio_yields_no_negatives(self, tmp_path):
        df = _synthetic_raw(50)
        path = _write_csv(tmp_path, df)
        prep = DataPreprocessor(negative_ratio=0.0, use_property_matched_decoys=False)
        result = prep.preprocess(path)
        assert (result["bound"] == 0).sum() == 0

    def test_keep_only_measured_filters_rows(self, tmp_path):
        df = _synthetic_raw(100)
        n_measured = (df["kiba_score_estimated"] == False).sum()  # noqa: E712
        path = _write_csv(tmp_path, df)
        prep = DataPreprocessor(keep_only_measured=True, use_property_matched_decoys=False,
                                negative_ratio=0.0)
        result = prep.preprocess(path)
        real = result[result["kiba_score"].notna()]
        assert len(real) <= n_measured
