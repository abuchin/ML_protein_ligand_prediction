"""Data loading, label creation, and negative sample generation for protein-ligand binding."""
from __future__ import annotations

import logging
from typing import Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class DataPreprocessor:
    """Clean the raw KIBA dataset and produce a balanced binary classification table.

    KIBA score semantics (He et al. 2017):
        Lower score = stronger binding affinity.
        Literature binding threshold: kiba_score < 12.1 → positive (bound=1).

    kiba_score_estimated column:
        True  = score was *imputed* (less reliable).
        False = score was directly *measured* (gold standard).
        Audit the distribution in 01_eda.ipynb before toggling keep_only_measured.

    Args:
        kiba_threshold: KIBA score boundary; rows below this are labeled positive.
        kiba_binder_is_below: If True, bound=1 when score < threshold (correct direction).
        keep_only_measured: If True, drop rows where kiba_score_estimated is True.
        negative_ratio: Number of negative samples per positive.
        use_property_matched_decoys: Use DUD-E / DEKOIS style decoys (preferred).
            Falls back to random shuffle when RDKit is unavailable or SMILES are missing.
        random_state: Seed for all random operations.
    """

    def __init__(
        self,
        kiba_threshold: float = 12.1,
        kiba_binder_is_below: bool = True,
        keep_only_measured: bool = False,
        negative_ratio: float = 1.0,
        use_property_matched_decoys: bool = True,
        random_state: int = 42,
    ) -> None:
        self.kiba_threshold = kiba_threshold
        self.kiba_binder_is_below = kiba_binder_is_below
        self.keep_only_measured = keep_only_measured
        self.negative_ratio = negative_ratio
        self.use_property_matched_decoys = use_property_matched_decoys
        self.rng = np.random.default_rng(random_state)
        self.random_state = random_state

    # ── Public API ────────────────────────────────────────────────────────────

    def preprocess(
        self,
        data_path: str,
        smiles_map: Optional[Dict[int, str]] = None,
    ) -> pd.DataFrame:
        """Full preprocessing pipeline. Returns combined positive+negative DataFrame.

        Args:
            data_path: Path to Drug_Discovery_dataset.csv.
            smiles_map: dict mapping pubchem_cid → SMILES. Required for property-matched
                decoys; if None, falls back to random shuffle negatives.
        """
        df = self._load_and_clean(data_path)
        df = self._handle_duplicates(df)
        df = self._create_labels(df)

        positives = df[df["bound"] == 1].copy()
        n_pos = len(positives)
        logger.info("Positive pairs after labeling: %d", n_pos)

        negatives = self._create_negatives(positives, df, smiles_map)
        n_neg = len(negatives)
        logger.info("Negative pairs generated: %d  (ratio %.2f)", n_neg, n_neg / max(n_pos, 1))

        combined = pd.concat([positives, negatives], ignore_index=True)
        combined = combined.sample(frac=1, random_state=self.random_state).reset_index(drop=True)
        logger.info("Combined dataset: %d rows  (positive rate %.3f)", len(combined), combined["bound"].mean())
        return combined

    # ── Private helpers ───────────────────────────────────────────────────────

    def _load_and_clean(self, data_path: str) -> pd.DataFrame:
        df = pd.read_csv(data_path)
        logger.info("Loaded %d rows from %s", len(df), data_path)

        # Audit the estimated flag before filtering — log both sides.
        if "kiba_score_estimated" in df.columns:
            est_counts = df["kiba_score_estimated"].value_counts()
            logger.info("kiba_score_estimated counts:\n%s", est_counts.to_string())
            est_mean = df.groupby("kiba_score_estimated")["kiba_score"].describe()
            logger.info("KIBA score stats by estimated flag:\n%s", est_mean.to_string())

            if self.keep_only_measured:
                # Keep only rows where score is directly measured (estimated == False)
                before = len(df)
                df = df[df["kiba_score_estimated"] == False]  # noqa: E712
                logger.info("Kept measured-only rows: %d → %d", before, len(df))

        df = df.dropna(subset=["UniProt_ID", "pubchem_cid", "kiba_score"])
        df["pubchem_cid"] = df["pubchem_cid"].astype(int)
        return df

    def _handle_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Average KIBA scores for duplicate (UniProt_ID, pubchem_cid) pairs."""
        before = len(df)
        df = df.groupby(["UniProt_ID", "pubchem_cid"], as_index=False)["kiba_score"].mean()
        logger.info("Duplicates collapsed: %d → %d rows", before, len(df))
        return df

    def _create_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create binary binding label.

        Lower KIBA = stronger binding → bound=1 when kiba_score < kiba_threshold.
        """
        if self.kiba_binder_is_below:
            df["bound"] = (df["kiba_score"] < self.kiba_threshold).astype(int)
        else:
            df["bound"] = (df["kiba_score"] > self.kiba_threshold).astype(int)

        pos_rate = df["bound"].mean()
        logger.info(
            "Labels: threshold=%.2f, binder_is_below=%s → positive rate=%.3f",
            self.kiba_threshold,
            self.kiba_binder_is_below,
            pos_rate,
        )
        if pos_rate > 0.9 or pos_rate < 0.01:
            logger.warning(
                "Positive rate %.3f is extreme — verify kiba_threshold and kiba_binder_is_below.",
                pos_rate,
            )
        return df

    def _create_negatives(
        self,
        positives: pd.DataFrame,
        all_data: pd.DataFrame,
        smiles_map: Optional[Dict[int, str]],
    ) -> pd.DataFrame:
        """Generate negative (non-binding) pairs.

        Strategy 1 (preferred): Property-matched decoys (DUD-E / DEKOIS style).
            For each binder ligand, find chemically dissimilar ligands (Tanimoto < 0.4)
            with similar physicochemical properties (MW, logP, charge, rotatable bonds).
        Strategy 2 (fallback): Random permutation of pubchem_cid values.
        """
        n_target = int(len(positives) * self.negative_ratio)
        known_positives = set(zip(positives["UniProt_ID"], positives["pubchem_cid"]))
        candidate_cids = all_data["pubchem_cid"].unique().tolist()

        if self.use_property_matched_decoys and smiles_map:
            negatives = self._property_matched_decoys(positives, known_positives, smiles_map, n_target)
            if negatives is not None and len(negatives) >= n_target // 2:
                return negatives
            logger.warning("Property-matched decoys insufficient; falling back to random negatives.")

        return self._random_negatives(positives, known_positives, candidate_cids, n_target)

    def _property_matched_decoys(
        self,
        positives: pd.DataFrame,
        known_positives: set,
        smiles_map: Dict[int, str],
        n_target: int,
    ) -> Optional[pd.DataFrame]:
        """DUD-E style: dissimilar topology, similar physicochemistry.

        For speed, samples up to 10k candidate CIDs and uses vectorized Tanimoto.
        """
        try:
            from rdkit import Chem, DataStructs
            from rdkit.Chem import AllChem, Descriptors
        except ImportError:
            logger.warning("RDKit not available; cannot generate property-matched decoys.")
            return None

        def _props(mol):
            return {
                "mw": Descriptors.MolWt(mol),
                "logp": Descriptors.MolLogP(mol),
                "charge": Chem.rdmolops.GetFormalCharge(mol),
                "rot": Descriptors.NumRotatableBonds(mol),
            }

        def _fp(mol):
            return AllChem.GetMorganFingerprintAsBitVect(mol, 2, 1024)

        # Pre-compute fingerprints + properties for the candidate pool (up to 10k)
        pool_cids = list(smiles_map.keys())
        if len(pool_cids) > 10_000:
            pool_cids = self.rng.choice(pool_cids, size=10_000, replace=False).tolist()

        pool_mols = {}
        pool_fps = {}
        pool_props = {}
        for cid in pool_cids:
            smi = smiles_map.get(cid)
            if not smi:
                continue
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue
            pool_mols[cid] = mol
            pool_fps[cid] = _fp(mol)
            pool_props[cid] = _props(mol)

        if not pool_fps:
            return None

        pool_cid_list = list(pool_fps.keys())
        pool_fp_list = [pool_fps[c] for c in pool_cid_list]

        rows = []
        for _, pos_row in positives.iterrows():
            uniprot = pos_row["UniProt_ID"]
            cid = int(pos_row["pubchem_cid"])
            smi = smiles_map.get(cid)
            if not smi:
                continue
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue
            query_fp = _fp(mol)
            query_props = _props(mol)

            # Vectorized Tanimoto against the entire pool
            tanimotos = DataStructs.BulkTanimotoSimilarity(query_fp, pool_fp_list)

            per_positive_target = max(1, int(self.negative_ratio))
            decoys_added = 0
            for i, (candidate_cid, tani) in enumerate(zip(pool_cid_list, tanimotos)):
                if decoys_added >= per_positive_target:
                    break
                if candidate_cid == cid:
                    continue
                if (uniprot, candidate_cid) in known_positives:
                    continue
                if tani >= 0.4:
                    continue
                cp = pool_props[candidate_cid]
                if (
                    abs(cp["mw"] - query_props["mw"]) > 25
                    or abs(cp["logp"] - query_props["logp"]) > 1.5
                    or cp["charge"] != query_props["charge"]
                    or abs(cp["rot"] - query_props["rot"]) > 2
                ):
                    continue
                rows.append({"UniProt_ID": uniprot, "pubchem_cid": candidate_cid,
                              "kiba_score": np.nan, "bound": 0})
                decoys_added += 1

        if not rows:
            return None

        return pd.DataFrame(rows)

    def _random_negatives(
        self,
        positives: pd.DataFrame,
        known_positives: set,
        candidate_cids: list,
        n_target: int,
    ) -> pd.DataFrame:
        """Seeded random shuffle negatives. Fast fallback."""
        rows = []
        attempts = 0
        max_attempts = n_target * 10
        shuffled_cids = self.rng.permutation(candidate_cids)
        cid_iter = iter(shuffled_cids.tolist() * 20)  # cycle

        for _, pos_row in positives.iterrows():
            uniprot = pos_row["UniProt_ID"]
            per_positive = max(1, int(self.negative_ratio))
            added = 0
            while added < per_positive and attempts < max_attempts:
                attempts += 1
                try:
                    cid = next(cid_iter)
                except StopIteration:
                    break
                if (uniprot, cid) in known_positives:
                    continue
                rows.append({"UniProt_ID": uniprot, "pubchem_cid": cid,
                              "kiba_score": np.nan, "bound": 0})
                added += 1

            if len(rows) >= n_target:
                break

        return pd.DataFrame(rows)
