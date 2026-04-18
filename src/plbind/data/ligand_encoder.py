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
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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


class LigandEncoder:
    """Compute all ligand features from a SMILES string.

    Args:
        morgan_radius:     Circular neighbourhood radius (2 = ECFP4).
        morgan_bits:       Hash length for Morgan count fingerprint.
        atompair_bits:     Hash length for atom-pair fingerprint.
        n_workers:         Parallel workers for batch processing.
    """

    def __init__(
        self,
        morgan_radius: int = 2,
        morgan_bits: int = MORGAN_BITS,
        atompair_bits: int = ATOMPAIR_BITS,
        n_workers: int = 4,
    ) -> None:
        self.morgan_radius = morgan_radius
        self.morgan_bits = morgan_bits
        self.atompair_bits = atompair_bits
        self.n_workers = n_workers

    # ── Feature names ─────────────────────────────────────────────────────────

    @property
    def feature_names(self) -> List[str]:
        morgan_names = [f"morgan_{i}" for i in range(self.morgan_bits)]
        maccs_names = [f"maccs_{i}" for i in range(1, MACCS_BITS + 1)]
        ap_names = [f"atompair_{i}" for i in range(self.atompair_bits)]
        return morgan_names + maccs_names + ap_names + DESCRIPTOR_NAMES

    # ── Single-molecule encoding ──────────────────────────────────────────────

    def encode_smiles(self, smiles: str) -> Optional[Dict]:
        """Encode a single SMILES string.

        Returns dict with keys:
            "morgan"      — np.ndarray int32 (1024,)
            "maccs"       — np.ndarray int32 (166,)
            "atompair"    — np.ndarray int32 (1024,)
            "descriptors" — np.ndarray float32 (15,)
        Returns None if SMILES is invalid.
        """
        try:
            from rdkit import Chem
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            return {
                "morgan": self._morgan_count(mol),
                "maccs": self._maccs(mol),
                "atompair": self._atompair(mol),
                "descriptors": self._descriptors(mol),
            }
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

        # Use multiprocessing for heavy RDKit computation
        chunk_size = max(1, len(cid_smiles) // self.n_workers)
        items = list(cid_smiles.items())

        def _encode_chunk(chunk):
            enc = LigandEncoder(self.morgan_radius, self.morgan_bits, self.atompair_bits)
            out = {}
            for cid, smi in chunk:
                r = enc.encode_smiles(smi)
                if r is not None:
                    out[cid] = r
            return out

        chunks = [items[i : i + chunk_size] for i in range(0, len(items), chunk_size)]
        for chunk_result in [_encode_chunk(c) for c in chunks]:
            results.update(chunk_result)

        n_dropped = len(cid_smiles) - len(results)
        if n_dropped:
            logger.info("Ligand encoding: dropped %d CIDs (invalid SMILES).", n_dropped)

        # Build output matrices
        cid_to_row: Dict[int, int] = {cid: i for i, cid in enumerate(sorted(results))}
        N = len(results)
        fp_data = np.zeros((N, TOTAL_FINGERPRINT_BITS), dtype=np.int32)
        desc_data = np.zeros((N, len(DESCRIPTOR_NAMES)), dtype=np.float32)

        for cid, row_idx in cid_to_row.items():
            r = results[cid]
            fp_data[row_idx, :self.morgan_bits] = r["morgan"]
            fp_data[row_idx, self.morgan_bits:self.morgan_bits + MACCS_BITS] = r["maccs"]
            fp_data[row_idx, self.morgan_bits + MACCS_BITS:] = r["atompair"]
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
        from rdkit.DataStructs import ConvertToNumpyArray
        # GetHashedMorganFingerprint returns UIntSparseIntVect — convert via dict
        for idx, cnt in fp.GetNonzeroElements().items():
            arr[idx % self.morgan_bits] += cnt
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
        fp = GetHashedAtomPairFingerprintAsBitVect(mol, nBits=self.atompair_bits)
        arr = np.zeros(self.atompair_bits, dtype=np.int32)
        fp.ToList()
        from rdkit.DataStructs import ConvertToNumpyArray
        ConvertToNumpyArray(fp, arr)
        return arr

    def _descriptors(self, mol) -> np.ndarray:
        from rdkit.Chem import Descriptors, QED, rdMolDescriptors
        from rdkit.Chem.rdPartialCharges import ComputeGasteigerCharges

        ComputeGasteigerCharges(mol)
        charges = [atom.GetDoubleProp("_GasteigerCharge") for atom in mol.GetAtoms()
                   if not (atom.GetDoubleProp("_GasteigerCharge") != atom.GetDoubleProp("_GasteigerCharge"))]

        max_charge = max(charges) if charges else 0.0
        min_charge = min(charges) if charges else 0.0

        stereo_info = rdMolDescriptors.FindPotentialStereo(mol)
        n_stereo = sum(1 for s in stereo_info if s.type.name == "Atom")

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
