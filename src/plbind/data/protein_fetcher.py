"""UniProt REST API client with disk caching and auxiliary feature extraction.

Fetches protein metadata (GO terms, Pfam families, subcellular location, organism,
sequence length) and one-hot encodes them into a 95-dimensional feature matrix.

These annotated features complement ESM-2 embeddings (which capture sequence context)
with human-curated biological function labels.

Reference:
    UniProt REST API: https://rest.uniprot.org/uniprotkb/{id}.json
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import requests

from plbind.utils.cache import disk_cache

logger = logging.getLogger(__name__)

_UNIPROT_REST = "https://rest.uniprot.org/uniprotkb/{uid}.json"
_SEQUENCE_REST = "https://rest.uniprot.org/uniprotkb/{uid}.fasta"


class UniProtFetcher:
    """Fetch and cache protein metadata and sequences from UniProt.

    Args:
        cache_dir: Directory for caching raw JSON responses and built features.
        top_go:    Number of most frequent GO molecular-function terms to encode.
        top_pfam:  Number of most frequent Pfam domain families to encode.
        request_delay: Seconds to sleep between API calls (be kind to UniProt).
    """

    def __init__(
        self,
        cache_dir: Path,
        top_go: int = 50,
        top_pfam: int = 30,
        request_delay: float = 0.1,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.top_go = top_go
        self.top_pfam = top_pfam
        self.request_delay = request_delay
        self._go_vocab: Optional[List[str]] = None
        self._pfam_vocab: Optional[List[str]] = None

    # ── Public API ────────────────────────────────────────────────────────────

    def fetch_sequence(self, uniprot_id: str) -> Optional[str]:
        """Fetch the canonical amino-acid sequence from UniProt (FASTA endpoint)."""
        cached = self._load_sequence_cache(uniprot_id)
        if cached is not None:
            return cached
        try:
            url = _SEQUENCE_REST.format(uid=uniprot_id)
            resp = requests.get(url, timeout=15)
            if resp.status_code != 200:
                logger.warning("UniProt sequence fetch failed for %s (HTTP %d)", uniprot_id, resp.status_code)
                return None
            time.sleep(self.request_delay)
            lines = resp.text.strip().splitlines()
            seq = "".join(l for l in lines if not l.startswith(">"))
            self._save_sequence_cache(uniprot_id, seq)
            return seq
        except Exception as exc:
            logger.warning("Exception fetching sequence for %s: %s", uniprot_id, exc)
            return None

    def fetch_metadata(self, uniprot_id: str) -> Dict:
        """Fetch structured metadata (GO, Pfam, location, organism) from UniProt JSON."""
        cached = self._load_metadata_cache(uniprot_id)
        if cached is not None:
            return cached
        try:
            url = _UNIPROT_REST.format(uid=uniprot_id)
            resp = requests.get(url, timeout=15, headers={"Accept": "application/json"})
            if resp.status_code != 200:
                logger.warning("Metadata fetch failed for %s (HTTP %d)", uniprot_id, resp.status_code)
                return {}
            time.sleep(self.request_delay)
            raw = resp.json()
            parsed = self._parse_metadata(raw)
            self._save_metadata_cache(uniprot_id, parsed)
            return parsed
        except Exception as exc:
            logger.warning("Exception fetching metadata for %s: %s", uniprot_id, exc)
            return {}

    def build_auxiliary_features(
        self, uniprot_ids: List[str], fit_vocab: bool = True
    ) -> pd.DataFrame:
        """Build one-hot + continuous auxiliary features for a list of UniProt IDs.

        Returns a DataFrame indexed by UniProt_ID with ~95 columns:
            - GO molecular function terms (top_go binary columns)
            - Pfam protein family (top_pfam binary columns)
            - Subcellular location (10 binary columns)
            - Log sequence length (1 continuous column)
            - Organism category (4 binary columns: human / mouse / rat / other)

        Args:
            uniprot_ids: Proteins to encode.
            fit_vocab:   If True (default), build GO/Pfam vocabularies from this list.
                         Set False for val/test proteins to avoid leaking their term
                         frequencies into the vocabulary used for training proteins.
        """
        logger.info("Fetching metadata for %d proteins (fit_vocab=%s)...", len(uniprot_ids), fit_vocab)
        meta: Dict[str, Dict] = {}
        for i, uid in enumerate(uniprot_ids):
            meta[uid] = self.fetch_metadata(uid)
            if (i + 1) % 50 == 0:
                logger.info("  Fetched %d / %d", i + 1, len(uniprot_ids))

        if fit_vocab:
            self._go_vocab = self._top_terms(meta, "go_mf", self.top_go)
            self._pfam_vocab = self._top_terms(meta, "pfam", self.top_pfam)
        elif self._go_vocab is None or self._pfam_vocab is None:
            logger.warning(
                "fit_vocab=False but vocabulary is not yet built; building from current proteins. "
                "Call build_auxiliary_features on training proteins first."
            )
            self._go_vocab = self._top_terms(meta, "go_mf", self.top_go)
            self._pfam_vocab = self._top_terms(meta, "pfam", self.top_pfam)

        rows = {}
        for uid in uniprot_ids:
            rows[uid] = self._encode_one(meta.get(uid, {}))
        df = pd.DataFrame.from_dict(rows, orient="index")
        df.index.name = "UniProt_ID"
        logger.info("Auxiliary features: %d proteins × %d features", len(df), df.shape[1])
        return df

    # ── Parsing ───────────────────────────────────────────────────────────────

    def _parse_metadata(self, raw: dict) -> dict:
        """Extract relevant fields from UniProt JSON response."""
        parsed: dict = {"go_mf": [], "pfam": [], "locations": [], "organism": "", "seq_len": 0}

        # GO molecular function terms
        for ref in raw.get("uniProtKBCrossReferences", []):
            if ref.get("database") == "GO":
                for prop in ref.get("properties", []):
                    if prop.get("key") == "GoTerm" and prop.get("value", "").startswith("F:"):
                        parsed["go_mf"].append(prop["value"][2:])  # strip "F:" prefix
            elif ref.get("database") == "Pfam":
                entry_name = ref.get("id", "")
                if entry_name:
                    parsed["pfam"].append(entry_name)

        # Subcellular location
        for comment in raw.get("comments", []):
            if comment.get("commentType") == "SUBCELLULAR LOCATION":
                for loc in comment.get("subcellularLocations", []):
                    loc_name = loc.get("location", {}).get("value", "")
                    if loc_name:
                        parsed["locations"].append(loc_name.lower())

        # Organism
        organism = raw.get("organism", {}).get("scientificName", "").lower()
        parsed["organism"] = organism

        # Sequence length
        parsed["seq_len"] = raw.get("sequence", {}).get("length", 0)

        return parsed

    def _encode_one(self, meta: dict) -> dict:
        """Encode a single protein's metadata into a feature dict."""
        features: dict = {}

        # GO molecular function
        go_terms = set(meta.get("go_mf", []))
        for term in (self._go_vocab or []):
            features[f"go_{term[:40]}"] = float(term in go_terms)

        # Pfam
        pfam_ids = set(meta.get("pfam", []))
        for pfam in (self._pfam_vocab or []):
            features[f"pfam_{pfam}"] = float(pfam in pfam_ids)

        # Subcellular location (10 canonical locations)
        locs = set(meta.get("locations", []))
        for loc in _CANONICAL_LOCATIONS:
            features[f"loc_{loc.replace(' ', '_')}"] = float(
                any(loc in l for l in locs)
            )

        # Organism (4 categories)
        organism = meta.get("organism", "").lower()
        features["org_human"] = float("homo sapiens" in organism)
        features["org_mouse"] = float("mus musculus" in organism)
        features["org_rat"] = float("rattus" in organism)
        features["org_other"] = float(
            not any(k in organism for k in ("homo sapiens", "mus musculus", "rattus"))
        )

        # Log sequence length
        seq_len = meta.get("seq_len", 0)
        features["log_seq_len"] = float(np.log1p(seq_len))

        return features

    @staticmethod
    def _top_terms(meta: Dict[str, dict], key: str, top_n: int) -> List[str]:
        from collections import Counter
        counter: Counter = Counter()
        for m in meta.values():
            for term in m.get(key, []):
                counter[term] += 1
        return [term for term, _ in counter.most_common(top_n)]

    # ── Disk cache helpers ────────────────────────────────────────────────────

    def _load_metadata_cache(self, uid: str) -> Optional[dict]:
        p = self.cache_dir / "metadata" / f"{uid}.pkl"
        if p.exists():
            import pickle
            with p.open("rb") as f:
                return pickle.load(f)
        return None

    def _save_metadata_cache(self, uid: str, data: dict) -> None:
        import pickle
        p = self.cache_dir / "metadata" / f"{uid}.pkl"
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("wb") as f:
            pickle.dump(data, f)

    def _load_sequence_cache(self, uid: str) -> Optional[str]:
        p = self.cache_dir / "sequences" / f"{uid}.txt"
        if p.exists():
            return p.read_text().strip()
        return None

    def _save_sequence_cache(self, uid: str, seq: str) -> None:
        p = self.cache_dir / "sequences" / f"{uid}.txt"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(seq)


# 10 canonical subcellular locations used for one-hot encoding
_CANONICAL_LOCATIONS = [
    "cytoplasm",
    "nucleus",
    "cell membrane",
    "endoplasmic reticulum",
    "mitochondrion",
    "golgi apparatus",
    "secreted",
    "lysosome",
    "peroxisome",
    "endosome",
]
