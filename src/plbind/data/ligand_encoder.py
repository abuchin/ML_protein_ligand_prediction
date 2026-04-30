"""Ligand feature encoder: Morgan count fingerprints + MACCS + atom-pair + physicochemical descriptors.

Feature blocks:
    1. Morgan radius-2 count fingerprint (1024-dim)  — circular substructure counts;
       counts are more informative than binary bits for frequent pharmacophores.
    2. MACCS keys (166-dim)  — pharmacophore-based structural keys encoding HBD/HBA
       topology, ring systems, charged groups that Morgan can miss.
    3. Atom-pair fingerprint (1024-dim)  — encodes inter-atom type distances;
       complementary to the neighbourhood-centric Morgan.
    4. Physicochemical descriptors (15 continuous features).

Fingerprint arrays are stored as scipy.sparse.csr_matrix to save ~8 GB RAM
at 500k rows compared with dense float32. XGBoost, LightGBM, and
LogisticRegression all accept sparse input directly.
"""
from __future__ import annotations

import logging
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import scipy.sparse as sp

logger = logging.getLogger(__name__)

# Names of the 15 physicochemical descriptors (for feature-importance labelling)
DESCRIPTOR_NAMES = [
    "MolecularWeight",
    "LogP",
    "TPSA",
    "NumRotatableBonds",
    "NumHDonors",
    "NumHAcceptors",
    "NumAromaticRings",
    "FractionCSP3",
    "BertzComplexity",
    "QED",
    "NumRings",
    "NumHeterocycles",
    "MaxPartialCharge",
    "MinPartialCharge",
    "NumStereocenters",
]

MORGAN_BITS = 1024
MACCS_BITS = 166  # keys 1–166; key 0 is unused and stripped
ATOMPAIR_BITS = 1024
TOTAL_FINGERPRINT_BITS = MORGAN_BITS + MACCS_BITS + ATOMPAIR_BITS  # 2214
TOTAL_FEATURES = TOTAL_FINGERPRINT_BITS + len(DESCRIPTOR_NAMES)    # 2229


def _encode_chunk_fn(args: Tuple[List, Dict[str, Any]]) -> Dict[int, Dict]:
    """Module-level worker for ProcessPoolExecutor (must be top-level to be picklable)."""
    chunk, params = args
    enc = LigandEncoder(**params)
    out: Dict[int, Dict] = {}
    for cid, smi in chunk:
        r = enc.encode_smiles(smi)
        if r is not None:
            out[cid] = r
    return out


class LigandEncoder:
    """Compute all ligand features from a SMILES string.

    Args:
        morgan_radius:     Circular neighbourhood radius (2 = ECFP4).
        morgan_bits:       Hash length for Morgan count fingerprint.
        atompair_bits:     Hash length for atom-pair fingerprint.
        use_maccs:         Include 166-bit MACCS pharmacophore keys.
        use_atompair:      Include 1024-bit atom-pair fingerprint.
        morgan_use_counts: Use count fingerprint; if False, binarizes to 0/1.
        n_workers:         Parallel workers for batch processing.
    """

    def __init__(
        self,
        morgan_radius: int = 2,
        morgan_bits: int = MORGAN_BITS,
        atompair_bits: int = ATOMPAIR_BITS,
        use_maccs: bool = True,
        use_atompair: bool = True,
        morgan_use_counts: bool = True,
        n_workers: int = 4,
    ) -> None:
        self.morgan_radius = morgan_radius
        self.morgan_bits = morgan_bits
        self.atompair_bits = atompair_bits
        self.use_maccs = use_maccs
        self.use_atompair = use_atompair
        self.morgan_use_counts = morgan_use_counts
        self.n_workers = n_workers

    @property
    def fp_dim(self) -> int:
        """Total fingerprint dimension based on enabled blocks."""
        dim = self.morgan_bits
        if self.use_maccs:
            dim += MACCS_BITS
        if self.use_atompair:
            dim += self.atompair_bits
        return dim

    # ── Feature names ─────────────────────────────────────────────────────────

    @property
    def feature_names(self) -> List[str]:
        names = [f"morgan_{i}" for i in range(self.morgan_bits)]
        if self.use_maccs:
            names += [f"maccs_{i}" for i in range(1, MACCS_BITS + 1)]
        if self.use_atompair:
            names += [f"atompair_{i}" for i in range(self.atompair_bits)]
        return names + DESCRIPTOR_NAMES

    # ── Single-molecule encoding ──────────────────────────────────────────────

    def encode_smiles(self, smiles: str) -> Optional[Dict]:
        """Encode a single SMILES string.

        Returns dict with keys (only enabled blocks are included):
            "morgan"      — np.ndarray int32 (morgan_bits,)
            "maccs"       — np.ndarray int32 (166,)   [if use_maccs]
            "atompair"    — np.ndarray int32 (atompair_bits,) [if use_atompair]
            "descriptors" — np.ndarray float32 (15,)
        Returns None if SMILES is invalid.
        """
        try:
            from rdkit import Chem
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            result: Dict = {"morgan": self._morgan_count(mol)}
            if self.use_maccs:
                result["maccs"] = self._maccs(mol)
            if self.use_atompair:
                result["atompair"] = self._atompair(mol)
            result["descriptors"] = self._descriptors(mol)
            return result
        except Exception as exc:
            logger.debug("encode_smiles error for '%s': %s", smiles[:40], exc)
            return None

    def encode_flat(self, smiles: str) -> Optional[np.ndarray]:
        """Encode a single SMILES to a flat float32 array of length TOTAL_FEATURES."""
        result = self.encode_smiles(smiles)
        if result is None:
            return None
        return np.concatenate([
            result["morgan"].astype(np.float32),
            result["maccs"].astype(np.float32),
            result["atompair"].astype(np.float32),
            result["descriptors"],
        ])

    # ── Batch encoding ────────────────────────────────────────────────────────

    def encode_batch(
        self,
        cid_smiles: Dict[int, str],
        cache_dir: Optional[Path] = None,
    ) -> Tuple[Dict[int, int], sp.csr_matrix, np.ndarray]:
        """Encode all (CID, SMILES) pairs.

        Returns:
            cid_to_row:   dict mapping pubchem_cid → row index in the output matrices.
            fp_matrix:    scipy.sparse.csr_matrix (N, TOTAL_FINGERPRINT_BITS) int32.
            desc_matrix:  np.ndarray (N, 15) float32.
        """
        if cache_dir is not None:
            cached = self._load_batch_cache(cache_dir)
            if cached is not None:
                logger.info("Loaded ligand features from cache (%d CIDs).", len(cached[0]))
                return cached

        logger.info("Encoding %d ligands...", len(cid_smiles))
        results: Dict[int, Dict] = {}

        chunk_size = max(1, len(cid_smiles) // max(self.n_workers, 1))
        items = list(cid_smiles.items())
        chunks = [items[i : i + chunk_size] for i in range(0, len(items), chunk_size)]
        encoder_params = {
            "morgan_radius": self.morgan_radius,
            "morgan_bits": self.morgan_bits,
            "atompair_bits": self.atompair_bits,
            "use_maccs": self.use_maccs,
            "use_atompair": self.use_atompair,
            "morgan_use_counts": self.morgan_use_counts,
        }
        work = [(c, encoder_params) for c in chunks]

        try:
            with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
                for chunk_result in executor.map(_encode_chunk_fn, work):
                    results.update(chunk_result)
        except Exception as exc:
            logger.warning("Multiprocessing failed (%s); falling back to sequential.", exc)
            results.clear()
            for item in work:
                results.update(_encode_chunk_fn(item))

        n_dropped = len(cid_smiles) - len(results)
        if n_dropped:
            logger.info("Ligand encoding: dropped %d CIDs (invalid SMILES).", n_dropped)

        # Build output matrices
        cid_to_row: Dict[int, int] = {cid: i for i, cid in enumerate(sorted(results))}
        N = len(results)
        fp_data = np.zeros((N, self.fp_dim), dtype=np.int32)
        desc_data = np.zeros((N, len(DESCRIPTOR_NAMES)), dtype=np.float32)

        for cid, row_idx in cid_to_row.items():
            r = results[cid]
            offset = 0
            fp_data[row_idx, offset:offset + self.morgan_bits] = r["morgan"]
            offset += self.morgan_bits
            if self.use_maccs:
                fp_data[row_idx, offset:offset + MACCS_BITS] = r["maccs"]
                offset += MACCS_BITS
            if self.use_atompair:
                fp_data[row_idx, offset:offset + self.atompair_bits] = r["atompair"]
            desc_data[row_idx] = r["descriptors"]

        fp_sparse = sp.csr_matrix(fp_data, dtype=np.int32)

        if cache_dir is not None:
            self._save_batch_cache(cache_dir, (cid_to_row, fp_sparse, desc_data))

        return cid_to_row, fp_sparse, desc_data

    # ── RDKit fingerprint methods ─────────────────────────────────────────────

    def _morgan_count(self, mol) -> np.ndarray:
        from rdkit.Chem import rdMolDescriptors
        fp = rdMolDescriptors.GetHashedMorganFingerprint(mol, self.morgan_radius, self.morgan_bits)
        arr = np.zeros(self.morgan_bits, dtype=np.int32)
        for idx, cnt in fp.GetNonzeroElements().items():
            arr[idx % self.morgan_bits] += cnt
        if not self.morgan_use_counts:
            arr = (arr > 0).astype(np.int32)
        return arr

    def _maccs(self, mol) -> np.ndarray:
        from rdkit.Chem import MACCSkeys
        fp = MACCSkeys.GenMACCSKeys(mol)
        arr = np.zeros(MACCS_BITS, dtype=np.int32)
        # Key 0 is always 0 (unused); keys 1–166 → indices 0–165
        for i in range(1, MACCS_BITS + 1):
            arr[i - 1] = int(fp[i])
        return arr

    def _atompair(self, mol) -> np.ndarray:
        from rdkit.Chem.rdMolDescriptors import GetHashedAtomPairFingerprintAsBitVect
        from rdkit.DataStructs import ConvertToNumpyArray
        fp = GetHashedAtomPairFingerprintAsBitVect(mol, nBits=self.atompair_bits)
        arr = np.zeros(self.atompair_bits, dtype=np.int32)
        ConvertToNumpyArray(fp, arr)
        return arr

    def _descriptors(self, mol) -> np.ndarray:
        from rdkit.Chem import Descriptors, QED
        from rdkit.Chem import rdMolDescriptors
        from rdkit.Chem.rdPartialCharges import ComputeGasteigerCharges

        ComputeGasteigerCharges(mol)
        charges = [atom.GetDoubleProp("_GasteigerCharge") for atom in mol.GetAtoms()]
        # Filter out NaN and inf produced by Gasteiger convergence failures
        charges = [c for c in charges if np.isfinite(c)]

        max_charge = float(np.clip(max(charges), -5, 5)) if charges else 0.0
        min_charge = float(np.clip(min(charges), -5, 5)) if charges else 0.0

        n_stereo = rdMolDescriptors.CalcNumAtomStereoCenters(mol)

        values = [
            Descriptors.MolWt(mol),
            Descriptors.MolLogP(mol),
            Descriptors.TPSA(mol),
            Descriptors.NumRotatableBonds(mol),
            Descriptors.NumHDonors(mol),
            Descriptors.NumHAcceptors(mol),
            Descriptors.NumAromaticRings(mol),
            Descriptors.FractionCSP3(mol),
            Descriptors.BertzCT(mol),
            QED.qed(mol),
            rdMolDescriptors.CalcNumRings(mol),
            rdMolDescriptors.CalcNumHeterocycles(mol),
            float(max_charge),
            float(min_charge),
            float(n_stereo),
        ]
        return np.array(values, dtype=np.float32)

    # ── Cache helpers ─────────────────────────────────────────────────────────

    def _load_batch_cache(self, cache_dir: Path):
        import pickle
        p = cache_dir / "ligand_features.pkl"
        if p.exists():
            with p.open("rb") as f:
                return pickle.load(f)
        return None

    def _save_batch_cache(self, cache_dir: Path, data) -> None:
        import pickle
        cache_dir.mkdir(parents=True, exist_ok=True)
        p = cache_dir / "ligand_features.pkl"
        with p.open("wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
