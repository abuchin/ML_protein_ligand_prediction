"""Tests for all split strategies in Splitter."""
import numpy as np
import pandas as pd
import pytest

from plbind.training.splitter import Splitter


def _make_df(n_proteins: int = 30, n_ligs_per_protein: int = 10, seed: int = 42) -> pd.DataFrame:
    """Synthetic DTI dataset with disjoint protein/ligand ID spaces."""
    rng = np.random.RandomState(seed)
    rows = []
    for p_idx in range(n_proteins):
        uid = f"P{p_idx:04d}"
        for l_idx in range(n_ligs_per_protein):
            cid = p_idx * n_ligs_per_protein + l_idx + 1
            bound = int(rng.random() < 0.4)
            rows.append({"UniProt_ID": uid, "pubchem_cid": cid, "bound": bound})
    df = pd.DataFrame(rows)
    # Guarantee at least one positive per stratum (needed for stratified splits)
    df.loc[0, "bound"] = 1
    df.loc[1, "bound"] = 0
    return df


class TestRandomSplit:
    def test_covers_all_rows(self):
        s = Splitter(test_size=0.2, val_size=0.1, random_state=42)
        df = _make_df()
        train, val, test = s.random_split(df)
        assert len(train) + len(val) + len(test) == len(df)

    def test_no_row_duplication(self):
        s = Splitter(random_state=42)
        train, val, test = s.random_split(_make_df())
        all_idx = list(train.index) + list(val.index) + list(test.index)
        assert len(all_idx) == len(set(all_idx))

    def test_test_fraction_approximate(self):
        s = Splitter(test_size=0.2, random_state=42)
        df = _make_df(n_proteins=50)
        train, val, test = s.random_split(df)
        assert abs(len(test) / len(df) - 0.2) < 0.05


class TestColdProteinSplit:
    def test_no_protein_overlap_train_test(self):
        s = Splitter(random_state=42)
        df = _make_df(n_proteins=40)
        train, val, test = s.cold_protein_split(df)
        assert len(set(train["UniProt_ID"]) & set(test["UniProt_ID"])) == 0

    def test_no_protein_overlap_val_test(self):
        s = Splitter(random_state=42)
        df = _make_df(n_proteins=40)
        train, val, test = s.cold_protein_split(df)
        assert len(set(val["UniProt_ID"]) & set(test["UniProt_ID"])) == 0

    def test_no_protein_overlap_train_val(self):
        s = Splitter(random_state=42)
        df = _make_df(n_proteins=40)
        train, val, test = s.cold_protein_split(df)
        assert len(set(train["UniProt_ID"]) & set(val["UniProt_ID"])) == 0

    def test_all_proteins_assigned(self):
        s = Splitter(random_state=42)
        df = _make_df(n_proteins=30)
        train, val, test = s.cold_protein_split(df)
        all_proteins = set(train["UniProt_ID"]) | set(val["UniProt_ID"]) | set(test["UniProt_ID"])
        assert all_proteins == set(df["UniProt_ID"])

    def test_test_protein_fraction_reasonable(self):
        s = Splitter(test_size=0.2, random_state=42)
        df = _make_df(n_proteins=40)
        train, val, test = s.cold_protein_split(df)
        frac = test["UniProt_ID"].nunique() / df["UniProt_ID"].nunique()
        assert 0.1 < frac < 0.4


class TestColdLigandSplit:
    def test_no_ligand_overlap_train_test(self):
        s = Splitter(random_state=42)
        df = _make_df(n_proteins=10, n_ligs_per_protein=20)
        train, val, test = s.cold_ligand_split(df)
        assert len(set(train["pubchem_cid"]) & set(test["pubchem_cid"])) == 0

    def test_no_ligand_overlap_val_test(self):
        s = Splitter(random_state=42)
        df = _make_df(n_proteins=10, n_ligs_per_protein=20)
        train, val, test = s.cold_ligand_split(df)
        assert len(set(val["pubchem_cid"]) & set(test["pubchem_cid"])) == 0

    def test_covers_all_ligands(self):
        s = Splitter(random_state=42)
        df = _make_df(n_proteins=10, n_ligs_per_protein=20)
        train, val, test = s.cold_ligand_split(df)
        all_ligs = (
            set(train["pubchem_cid"]) | set(val["pubchem_cid"]) | set(test["pubchem_cid"])
        )
        assert all_ligs == set(df["pubchem_cid"])


class TestColdBothSplit:
    def test_no_protein_overlap_train_test(self):
        s = Splitter(random_state=42)
        df = _make_df(n_proteins=40)
        train, val, test = s.cold_both_split(df)
        assert len(set(train["UniProt_ID"]) & set(test["UniProt_ID"])) == 0

    def test_no_known_binder_cids_in_test(self):
        """Test proteins unseen; binder CIDs seen in training must not appear in test."""
        s = Splitter(random_state=42)
        df = _make_df(n_proteins=40)
        train, val, test = s.cold_both_split(df)
        train_binder_cids = set(train.loc[train["bound"] == 1, "pubchem_cid"])
        val_binder_cids = set(val.loc[val["bound"] == 1, "pubchem_cid"])
        known_binders = train_binder_cids | val_binder_cids
        assert len(known_binders & set(test["pubchem_cid"])) == 0


class TestDispatch:
    @pytest.mark.parametrize("strategy", ["random", "cold_protein", "cold_ligand", "cold_both"])
    def test_dispatch_returns_three_dataframes(self, strategy):
        s = Splitter(random_state=42)
        result = s.split(_make_df(n_proteins=40), strategy)
        assert len(result) == 3
        for part in result:
            assert isinstance(part, pd.DataFrame)

    def test_unknown_strategy_raises(self):
        s = Splitter()
        with pytest.raises(ValueError, match="Unknown split strategy"):
            s.split(_make_df(), "not_a_strategy")

    def test_split_sizes_sum_to_total(self):
        for strategy in ("random", "cold_protein", "cold_ligand"):
            s = Splitter(random_state=0)
            df = _make_df(n_proteins=40)
            train, val, test = s.split(df, strategy)
            assert len(train) + len(val) + len(test) == len(df), strategy
