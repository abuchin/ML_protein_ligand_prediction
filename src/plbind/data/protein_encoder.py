"""ESM-2 protein language model encoder with mean+max pooling and disk caching.

Why ESM-2 over ProtBERT (the current encoder):
  - Trained on UniRef50/90 (250M sequences, curated) vs. BFD's noisier corpus.
  - 4-8% better on FLIP downstream benchmarks (Lin et al. 2023).
  - Native HuggingFace support — no fair-esm package needed.
  - Mean+max pooling retains residue-level peak activations lost by mean-only pooling.

Model choices by hardware:
  - GPU available:  facebook/esm2_t33_650M_UR50D  → 1280-dim (2560 with mean_max)
  - CPU only:       facebook/esm2_t12_35M_UR50D   → 480-dim  (960 with mean_max)

Reference:
    Lin et al. (2023). "Evolutionary-scale prediction of atomic-level protein structure
    with a language model." Science 379, 1123-1130.
"""
from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)


class ESM2Encoder:
    """Encode protein sequences with ESM-2 and return pooled embeddings.

    Args:
        model_name:   HuggingFace model identifier.
        device:       "auto" | "cpu" | "cuda" | "mps"
        max_length:   Token limit (ESM-2 positional encoding cap is 1022).
        batch_size:   Number of sequences to process at once.
        pooling:      "mean" | "max" | "mean_max"
            - "mean":     Mean over non-special residue positions → embed_dim features.
            - "max":      Max  over non-special residue positions → embed_dim features.
            - "mean_max": Concatenate mean and max → 2 × embed_dim features.
        cache_dir:    If provided, embeddings are cached by (uniprot_id, model_name).
    """

    def __init__(
        self,
        model_name: str = "facebook/esm2_t33_650M_UR50D",
        device: str = "auto",
        max_length: int = 1022,
        batch_size: int = 8,
        pooling: str = "mean_max",
        cache_dir: Optional[Path] = None,
    ) -> None:
        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size
        self.pooling = pooling
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self._model = None
        self._tokenizer = None
        self._device: Optional[torch.device] = None

        if device == "auto":
            if torch.cuda.is_available():
                self._device = torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self._device = torch.device("mps")
            else:
                self._device = torch.device("cpu")
        else:
            self._device = torch.device(device)

        logger.info("ESM2Encoder: model=%s  device=%s  pooling=%s", model_name, self._device, pooling)

    @property
    def embed_dim(self) -> int:
        """Output dimension before pooling concatenation."""
        # ESM-2 embed dims: 35M→480, 150M→640, 650M→1280, 3B→2560
        dim_map = {
            "esm2_t6_8M": 320, "esm2_t12_35M": 480, "esm2_t30_150M": 640,
            "esm2_t33_650M": 1280, "esm2_t36_3B": 2560, "esm2_t48_15B": 5120,
        }
        for key, dim in dim_map.items():
            if key in self.model_name:
                return dim
        return 1280  # safe default

    @property
    def output_dim(self) -> int:
        """Final output dimension (accounting for pooling strategy)."""
        return self.embed_dim * (2 if self.pooling == "mean_max" else 1)

    def encode_batch(self, sequences: List[str]) -> np.ndarray:
        """Encode a list of sequences, return array of shape (N, output_dim)."""
        self._ensure_loaded()
        embeddings = []
        for i in range(0, len(sequences), self.batch_size):
            batch = sequences[i : i + self.batch_size]
            emb = self._encode_batch_raw(batch)
            embeddings.append(emb)
        return np.vstack(embeddings).astype(np.float32)

    def encode_with_cache(
        self,
        uniprot_ids: List[str],
        sequences: List[str],
    ) -> Dict[str, np.ndarray]:
        """Encode proteins, using disk cache when available.

        Returns dict mapping UniProt_ID → embedding array (float32).
        """
        results: Dict[str, np.ndarray] = {}
        to_compute: List[tuple] = []

        for uid, seq in zip(uniprot_ids, sequences):
            cached = self._load_cache(uid)
            if cached is not None:
                results[uid] = cached
            else:
                to_compute.append((uid, seq))

        if to_compute:
            logger.info("Computing ESM-2 embeddings for %d proteins (cache misses)...", len(to_compute))
            uids, seqs = zip(*to_compute)
            batch_embs = self.encode_batch(list(seqs))
            for uid, emb in zip(uids, batch_embs):
                results[uid] = emb
                self._save_cache(uid, emb)

        return results

    # ── Private ───────────────────────────────────────────────────────────────

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        from transformers import AutoTokenizer, EsmModel
        logger.info("Loading ESM-2 model: %s (this may take a moment)...", self.model_name)
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = EsmModel.from_pretrained(self.model_name)
        self._model.eval().to(self._device)
        logger.info("ESM-2 loaded on %s. Output dim: %d", self._device, self.output_dim)

    def _encode_batch_raw(self, sequences: List[str]) -> np.ndarray:
        """Run ESM-2 forward pass and pool hidden states."""
        inputs = self._tokenizer(
            sequences,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length + 2,  # +2 for [CLS] and [EOS]
        ).to(self._device)

        with torch.no_grad():
            outputs = self._model(**inputs)

        # last_hidden_state: (B, L, D) where L includes [CLS] and [EOS] tokens
        hidden = outputs.last_hidden_state  # (B, L, D)

        # Build mask that excludes [CLS] (pos 0), [EOS] (last real token), and [PAD]
        # attention_mask: 1 for real tokens (including [CLS]/[EOS]), 0 for [PAD]
        attention_mask = inputs["attention_mask"]  # (B, L)
        # Exclude first ([CLS]) and identify [EOS] as last non-pad token per row
        seq_lengths = attention_mask.sum(dim=1)  # (B,)

        # Create residue-only mask: exclude index 0 ([CLS]) and last real token ([EOS])
        residue_mask = attention_mask.clone().float()
        residue_mask[:, 0] = 0  # mask [CLS]
        for b_idx, length in enumerate(seq_lengths):
            residue_mask[b_idx, length - 1] = 0  # mask [EOS]

        residue_mask = residue_mask.unsqueeze(-1)  # (B, L, 1) for broadcasting

        masked_hidden = hidden * residue_mask  # zero out [CLS], [EOS], [PAD]
        residue_counts = residue_mask.squeeze(-1).sum(dim=1, keepdim=True).clamp(min=1)  # (B, 1)

        if self.pooling == "mean":
            pooled = masked_hidden.sum(dim=1) / residue_counts  # (B, D)
        elif self.pooling == "max":
            # Replace masked positions with large negative before max
            large_neg = -1e9 * (1 - residue_mask)
            pooled = (hidden + large_neg).max(dim=1).values  # (B, D)
        else:  # mean_max
            mean_pool = masked_hidden.sum(dim=1) / residue_counts
            large_neg = -1e9 * (1 - residue_mask)
            max_pool = (hidden + large_neg).max(dim=1).values
            pooled = torch.cat([mean_pool, max_pool], dim=1)  # (B, 2D)

        return pooled.cpu().float().numpy()

    def _cache_path(self, uniprot_id: str) -> Optional[Path]:
        if self.cache_dir is None:
            return None
        model_tag = self.model_name.replace("/", "_")
        return self.cache_dir / "esm2" / f"{uniprot_id}_{model_tag}_{self.pooling}.npy"

    def _load_cache(self, uniprot_id: str) -> Optional[np.ndarray]:
        p = self._cache_path(uniprot_id)
        if p is not None and p.exists():
            return np.load(p)
        return None

    def _save_cache(self, uniprot_id: str, emb: np.ndarray) -> None:
        p = self._cache_path(uniprot_id)
        if p is not None:
            p.parent.mkdir(parents=True, exist_ok=True)
            np.save(p, emb)
