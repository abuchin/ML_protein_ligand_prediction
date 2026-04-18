"""Train/val/test split strategies for protein-ligand binding datasets.

Four strategies are implemented to provide an honest picture of generalization:
  - random:        Rows split uniformly at random (optimistic upper bound).
  - cold_protein:  No UniProt_ID appears in both train and test.
  - cold_ligand:   No pubchem_cid appears in both train and test.
  - scaffold:      Bemis-Murcko scaffolds are held out (MoleculeNet benchmark style).
  - cold_both:     Both proteins and ligands are unseen in test (hardest / most realistic).

Always report all four; the gap between random and cold_both is the honest estimate of
how well the model generalizes to truly new protein-molecule pairs in drug discovery.
"""
from __future__ import annotations

import logging
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold, StratifiedKFold

logger = logging.getLogger(__name__)

_SPLIT = Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]  # train, val, test


class Splitter:
    """Implements multiple splitting strategies for DTI datasets.

    Args:
        test_size:    Fraction of data (or proteins/scaffolds) held out as test.
        val_size:     Fraction of *training* data held out as validation.
        random_state: Seed for reproducibility.
    """

    def __init__(
        self,
        test_size: float = 0.2,
        val_size: float = 0.1,
        random_state: int = 42,
    ) -> None:
        self.test_size = test_size
        self.val_size = val_size
        self.random_state = random_state
        self._rng = np.random.default_rng(random_state)

    # ── Public dispatch ───────────────────────────────────────────────────────

    def split(self, df: pd.DataFrame, strategy: str, **kwargs) -> _SPLIT:
        """Dispatch to the requested split strategy.

        Args:
            df:       DataFrame with columns UniProt_ID, pubchem_cid, bound.
            strategy: One of random | cold_protein | cold_ligand | scaffold | cold_both.
        """
        methods = {
            "random": self.random_split,
            "cold_protein": self.cold_protein_split,
            "cold_ligand": self.cold_ligand_split,
            "scaffold": self.scaffold_split,
            "cold_both": self.cold_both_split,
        }
        if strategy not in methods:
            raise ValueError(f"Unknown split strategy '{strategy}'. Choose from: {list(methods)}")
        train, val, test = methods[strategy](df, **kwargs)
        self._log_split(train, val, test, strategy)
        return train, val, test

    def get_cv_splitter(self, strategy: str, n_splits: int = 5):
        """Return a scikit-learn CV splitter compatible with the given strategy.

        Usage:
            cv = splitter.get_cv_splitter("cold_protein")
            for train_idx, val_idx in cv.split(X, y, groups=df["UniProt_ID"]):
                ...
        """
        if strategy == "random":
            return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)
        return StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)

    # ── Split strategies ──────────────────────────────────────────────────────

    def random_split(self, df: pd.DataFrame, **_) -> _SPLIT:
        """Stratified random row split (canonical baseline — optimistic)."""
        from sklearn.model_selection import train_test_split

        train_val, test = train_test_split(
            df, test_size=self.test_size, stratify=df["bound"], random_state=self.random_state
        )
        train, val = train_test_split(
            train_val,
            test_size=self.val_size / (1 - self.test_size),
            stratify=train_val["bound"],
            random_state=self.random_state,
        )
        return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)

    def cold_protein_split(self, df: pd.DataFrame, **_) -> _SPLIT:
        """No UniProt_ID appears in more than one partition.

        Proteins are stratified by their positive-interaction rate before splitting
        so each split has a similar distribution of "hard" vs "easy" proteins.
        """
        proteins = self._stratify_entities(df, group_col="UniProt_ID")
        test_proteins, train_val_proteins = self._entity_split(proteins, self.test_size)
        train_val_df = proteins[proteins["UniProt_ID"].isin(train_val_proteins)]
        val_proteins, train_proteins = self._entity_split(train_val_df, self.val_size)

        test = df[df["UniProt_ID"].isin(test_proteins)]
        val = df[df["UniProt_ID"].isin(val_proteins)]
        train = df[df["UniProt_ID"].isin(train_proteins)]
        return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)

    def cold_ligand_split(self, df: pd.DataFrame, **_) -> _SPLIT:
        """No pubchem_cid appears in more than one partition."""
        ligands = self._stratify_entities(df, group_col="pubchem_cid")
        test_ligs, train_val_ligs = self._entity_split(ligands, self.test_size)
        train_val_df = ligands[ligands["pubchem_cid"].isin(train_val_ligs)]
        val_ligs, train_ligs = self._entity_split(train_val_df, self.val_size)

        test = df[df["pubchem_cid"].isin(test_ligs)]
        val = df[df["pubchem_cid"].isin(val_ligs)]
        train = df[df["pubchem_cid"].isin(train_ligs)]
        return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)

    def scaffold_split(self, df: pd.DataFrame, smiles_map: Optional[Dict[int, str]] = None, **_) -> _SPLIT:
        """Bemis-Murcko scaffold split — standard MoleculeNet benchmark.

        Ligands sharing a scaffold travel together into the same partition.
        Falls back to cold_ligand_split if RDKit or smiles_map is unavailable.
        """
        scaffolds = self._compute_scaffolds(df, smiles_map)
        if scaffolds is None:
            logger.warning("Scaffold split unavailable; falling back to cold_ligand_split.")
            return self.cold_ligand_split(df)

        df = df.copy()
        df["_scaffold"] = df["pubchem_cid"].map(scaffolds).fillna("__unknown__")

        scaffold_entities = self._stratify_entities(df, group_col="_scaffold")
        test_sc, train_val_sc = self._entity_split(scaffold_entities, self.test_size)
        train_val_df = scaffold_entities[scaffold_entities["_scaffold"].isin(train_val_sc)]
        val_sc, train_sc = self._entity_split(train_val_df, self.val_size)

        test = df[df["_scaffold"].isin(test_sc)].drop(columns="_scaffold")
        val = df[df["_scaffold"].isin(val_sc)].drop(columns="_scaffold")
        train = df[df["_scaffold"].isin(train_sc)].drop(columns="_scaffold")
        return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)

    def cold_both_split(self, df: pd.DataFrame, **kwargs) -> _SPLIT:
        """Both proteins and ligands unseen in test — hardest and most realistic.

        Approximation: cold_protein_split first, then remove any CIDs from test
        that also appear in train. This reduces test size but maintains the guarantee.
        """
        train, val, test = self.cold_protein_split(df, **kwargs)
        train_cids = set(train["pubchem_cid"]) | set(val["pubchem_cid"])
        test = test[~test["pubchem_cid"].isin(train_cids)]
        logger.info("cold_both: test reduced to %d rows after removing shared CIDs.", len(test))
        return train, val, test.reset_index(drop=True)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _stratify_entities(self, df: pd.DataFrame, group_col: str) -> pd.DataFrame:
        """Build a per-entity summary with positive rate for stratified sampling."""
        agg = df.groupby(group_col)["bound"].agg(["mean", "count"]).reset_index()
        agg.columns = [group_col, "pos_rate", "n_rows"]
        # Bin positive rate into quartiles for stratification
        agg["stratum"] = pd.qcut(agg["pos_rate"], q=4, labels=False, duplicates="drop")
        return agg

    def _entity_split(
        self, entity_df: pd.DataFrame, split_frac: float
    ) -> Tuple[set, set]:
        """Split entity-level DataFrame; returns (held_out_set, remaining_set).

        Stratifies by stratum column to preserve positive-rate distribution.
        """
        from sklearn.model_selection import train_test_split

        group_col = entity_df.columns[0]
        # train_test_split: first output = 1-split_frac (remaining), second = split_frac (held-out)
        remaining_df, held_out_df = train_test_split(
            entity_df,
            test_size=split_frac,
            stratify=entity_df["stratum"] if entity_df["stratum"].nunique() > 1 else None,
            random_state=self.random_state,
        )
        return set(held_out_df[group_col]), set(remaining_df[group_col])

    def _compute_scaffolds(
        self, df: pd.DataFrame, smiles_map: Optional[Dict[int, str]]
    ) -> Optional[Dict[int, str]]:
        """Return dict mapping pubchem_cid → Bemis-Murcko SMILES scaffold."""
        if smiles_map is None:
            return None
        try:
            from rdkit import Chem
            from rdkit.Chem.Scaffolds import MurckoScaffold
        except ImportError:
            return None

        scaffolds: Dict[int, str] = {}
        for cid, smi in smiles_map.items():
            mol = Chem.MolFromSmiles(smi) if smi else None
            if mol is None:
                scaffolds[cid] = "__no_scaffold__"
            else:
                try:
                    sc_mol = MurckoScaffold.GetScaffoldForMol(mol)
                    scaffolds[cid] = Chem.MolToSmiles(sc_mol) if sc_mol else "__no_scaffold__"
                except Exception:
                    scaffolds[cid] = "__no_scaffold__"
        return scaffolds

    def _log_split(
        self, train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame, strategy: str
    ) -> None:
        def _summary(part: pd.DataFrame, name: str) -> str:
            pr = part["bound"].mean() if len(part) > 0 else float("nan")
            return f"{name}: {len(part)} rows, pos_rate={pr:.3f}"

        logger.info(
            "Split [%s]: %s | %s | %s",
            strategy,
            _summary(train, "train"),
            _summary(val, "val"),
            _summary(test, "test"),
        )
        # Verify no leakage for protein-aware splits
        if strategy in ("cold_protein", "cold_both"):
            overlap = set(train["UniProt_ID"]) & set(test["UniProt_ID"])
            if overlap:
                logger.error("Protein leakage detected: %d proteins in both train and test!", len(overlap))
            else:
                logger.info("Protein leakage check: OK (no shared proteins in train/test)")
