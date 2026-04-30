#!/usr/bin/env python3
"""Data preparation pipeline: preprocess → fetch sequences → encode features → save.

Usage:
    python scripts/run_data_prep.py
    python scripts/run_data_prep.py --n_samples 500      # quick dev run
    python scripts/run_data_prep.py --keep_measured      # use measured scores only
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Make plbind importable without installation (development mode)
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from plbind.config import CFG
from plbind.utils.logging import setup_logging
from plbind.utils.seed import set_all_seeds


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Protein-ligand data preparation")
    parser.add_argument("--n_samples", type=int, default=None, help="Subsample rows for quick dev runs")
    parser.add_argument("--n_proteins", type=int, default=None, help="Randomly sample N proteins (keeps all their rows)")
    parser.add_argument("--keep_measured", action="store_true", help="Keep only kiba_score_estimated==False")
    parser.add_argument("--no_decoys", action="store_true", help="Use random negatives instead of property-matched")
    parser.add_argument("--log_level", default="INFO")
    return parser.parse_args()


def _fetch_missing_smiles(combined: "pd.DataFrame", smiles_map: dict, cache_path: "Path") -> dict:
    """Fetch SMILES from PubChem for any CIDs not already in smiles_map.

    Uses the PubChem REST bulk property endpoint (200 CIDs per request).
    Sleeps 0.3s between requests to respect the rate limit (~5 req/s).
    Updates smiles_map in place and saves the updated cache to disk.
    """
    import time
    import urllib.request
    import json as _json
    import pickle

    logger = logging.getLogger(__name__)

    all_cids = set(combined["pubchem_cid"].dropna().astype(int))
    missing = sorted(all_cids - set(smiles_map.keys()))
    if not missing:
        logger.info("SMILES cache is complete — no new CIDs to fetch.")
        return smiles_map

    logger.info("Fetching SMILES for %d new CIDs from PubChem...", len(missing))
    batch_size = 200
    fetched = 0
    failed = 0

    for i in range(0, len(missing), batch_size):
        batch = missing[i : i + batch_size]
        cid_str = ",".join(str(c) for c in batch)
        url = (
            "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/"
            f"{cid_str}/property/ConnectivitySMILES/JSON"
        )
        try:
            with urllib.request.urlopen(url, timeout=30) as resp:
                data = _json.loads(resp.read())
            for prop in data.get("PropertyTable", {}).get("Properties", []):
                cid = int(prop["CID"])
                smi = prop.get("ConnectivitySMILES") or prop.get("CanonicalSMILES") or prop.get("IsomericSMILES")
                if smi:
                    smiles_map[cid] = smi
                    fetched += 1
        except Exception as exc:
            logger.debug("PubChem batch %d failed: %s", i // batch_size, exc)
            failed += len(batch)
        time.sleep(0.3)

        if (i // batch_size) % 20 == 0 and i > 0:
            logger.info("  ... %d / %d CIDs fetched so far", i + len(batch), len(missing))

    logger.info("SMILES fetch complete: %d fetched, %d failed.", fetched, failed)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("wb") as f:
        pickle.dump(smiles_map, f, protocol=pickle.HIGHEST_PROTOCOL)
    logger.info("Updated SMILES cache saved → %s (%d total entries)", cache_path, len(smiles_map))
    return smiles_map


def main() -> None:
    args = parse_args()
    setup_logging(CFG.outputs_dir / "logs", level=args.log_level)
    set_all_seeds(CFG.random_seed)
    logger = logging.getLogger(__name__)

    logger.info("=== Data Preparation START ===")
    logger.info("KIBA threshold: %.1f  (binder_is_below=%s)", CFG.kiba_threshold, CFG.kiba_binder_is_below)

    # ── 1. Preprocess ──────────────────────────────────────────────────────────
    from plbind.data.preprocessor import DataPreprocessor

    prep = DataPreprocessor(
        kiba_threshold=CFG.kiba_threshold,
        kiba_binder_is_below=CFG.kiba_binder_is_below,
        keep_only_measured=args.keep_measured,
        negative_ratio=CFG.negative_ratio,
        use_property_matched_decoys=(not args.no_decoys),
        random_state=CFG.random_seed,
    )

    # Load SMILES map for property-matched decoys (if available)
    smiles_map = None
    cid_smiles_path = CFG.data_dir / "processed" / "cid_to_smiles.pkl"
    if cid_smiles_path.exists():
        import pickle
        with cid_smiles_path.open("rb") as f:
            smiles_map = pickle.load(f)
        logger.info("Loaded SMILES map: %d entries", len(smiles_map))

    combined = prep.preprocess(str(CFG.raw_data_path), smiles_map=smiles_map)
    if args.n_proteins:
        rng = __import__("numpy").random.default_rng(CFG.random_seed)
        proteins = combined["UniProt_ID"].unique()
        sampled = rng.choice(proteins, size=min(args.n_proteins, len(proteins)), replace=False)
        combined = combined[combined["UniProt_ID"].isin(sampled)].reset_index(drop=True)
        logger.info("Filtered to %d proteins → %d rows.", len(sampled), len(combined))
    if args.n_samples:
        combined = combined.sample(n=min(args.n_samples, len(combined)),
                                   random_state=CFG.random_seed).reset_index(drop=True)
        logger.info("Subsampled to %d rows.", len(combined))

    CFG.processed_dir.mkdir(parents=True, exist_ok=True)
    combined.to_csv(CFG.processed_dir / "combined_data.csv", index=False)
    logger.info("Saved combined_data.csv (%d rows)", len(combined))

    # ── 1b. Fetch SMILES for any CIDs not already cached ─────────────────────
    smiles_map = _fetch_missing_smiles(combined, smiles_map or {}, cid_smiles_path)

    # ── 2. Fetch UniProt sequences and auxiliary metadata ─────────────────────
    from plbind.data.protein_fetcher import UniProtFetcher

    uniprot_ids = combined["UniProt_ID"].unique().tolist()
    logger.info("Fetching data for %d unique proteins...", len(uniprot_ids))

    fetcher = UniProtFetcher(cache_dir=CFG.cache_dir)
    sequences: dict = {}
    for uid in uniprot_ids:
        seq = fetcher.fetch_sequence(uid)
        if seq:
            sequences[uid] = seq
    logger.info("Sequences fetched: %d / %d", len(sequences), len(uniprot_ids))

    aux_features = fetcher.build_auxiliary_features(uniprot_ids)
    aux_features.to_csv(CFG.processed_dir / "aux_features.csv")
    logger.info("Auxiliary features saved.")

    # ── 3. Encode protein sequences with ESM-2 ────────────────────────────────
    from plbind.data.protein_encoder import ESM2Encoder

    encoder = ESM2Encoder(
        model_name=CFG.protein_encoder,
        pooling=CFG.protein_pooling,
        cache_dir=CFG.cache_dir,
        max_length=CFG.protein_max_length,
    )
    uids_with_seq = [uid for uid in uniprot_ids if uid in sequences]
    seqs_to_encode = [sequences[uid] for uid in uids_with_seq]
    protein_embeddings = encoder.encode_with_cache(uids_with_seq, seqs_to_encode)

    import pickle
    with (CFG.processed_dir / "protein_embeddings.pkl").open("wb") as f:
        pickle.dump(protein_embeddings, f)
    logger.info("Protein embeddings saved: %d proteins, dim=%d",
                len(protein_embeddings), encoder.output_dim)

    # ── 4. Encode ligands ─────────────────────────────────────────────────────
    from plbind.data.ligand_encoder import LigandEncoder

    if smiles_map is None:
        logger.warning("No SMILES map found. Skipping ligand encoding.")
        return

    cid_list = combined["pubchem_cid"].unique().tolist()
    cid_smiles_subset = {cid: smiles_map[cid] for cid in cid_list if cid in smiles_map}

    lig_encoder = LigandEncoder(
        morgan_radius=CFG.morgan_radius,
        morgan_bits=CFG.morgan_bits,
        use_maccs=CFG.use_maccs,
        use_atompair=CFG.use_atompair,
        morgan_use_counts=CFG.morgan_use_counts,
    )
    cid_to_row, fp_matrix, desc_matrix = lig_encoder.encode_batch(
        cid_smiles_subset, cache_dir=CFG.processed_dir
    )

    import scipy.sparse as sp
    sp.save_npz(CFG.processed_dir / "ligand_fp.npz", fp_matrix)
    import numpy as np
    np.save(CFG.processed_dir / "ligand_desc.npy", desc_matrix)
    with (CFG.processed_dir / "cid_to_row.pkl").open("wb") as f:
        pickle.dump(cid_to_row, f)
    logger.info("Ligand features saved: %d CIDs", len(cid_to_row))

    logger.info("=== Data Preparation DONE ===")


if __name__ == "__main__":
    main()
