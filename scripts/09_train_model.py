#!/usr/bin/env python3
"""
Script 09 – Train PyTorch binding-score predictor (research-grade v2).

Architecture upgrade (Section 6.3)
────────────────────────────────────
Input: one-hot encoded RNA sequence (4 × max_len) + structural feature vector

Stage 1 — Sequence encoder (1D CNN)
  Conv1d(4 → hidden) → ReLU → Conv1d(hidden → hidden*2) → ReLU

Stage 2 — Self-attention over CNN feature map (Section 6.3)
  MultiheadAttention on the temporal dimension of the CNN output.
  This allows the model to identify positional patterns (e.g. a particular
  subsequence that strongly correlates with binding) regardless of position.
  Implementation: nn.MultiheadAttention (native PyTorch ≥1.9, no extra libs).

  APPROXIMATION: self-attention trained on 50-500 sequences provides only
  weak positional learning.  It functions primarily as a learned global
  pooling with position-dependent weighting.

Stage 3 — Feature fusion MLP
  [global-max-pooled CNN features ‖ attention-pooled features
   ‖ structural feature vector] → Linear → ReLU → Dropout → Linear(1)

Feature set (Section 6.2)
──────────────────────────
• Sequence-derived: encoded as one-hot (stages 1-2)
• Structural features (concatenated to MLP input):
    gc_content, mfe, n_stems, n_pairs, n_loops,
    stem_fraction, loop_fraction
  These are normalised separately to [0, 1] across the training set.

Dataset (Section 6.1)
──────────────────────
All ranked CSVs from all SELEX rounds are merged (deduplication by sequence).
score_detailed CSVs are used to extract structural features; merged on seq_id.

Use case (Section 6.4)
───────────────────────
The trained model is used ONLY for pre-screening: the top prescreening_top_fraction
sequences in a new pool are forwarded to docking; others are discarded.
This is enforced in 08_selex_iteration.py.

Dependencies: PyTorch ≥ 2.0, numpy, pandas, yaml
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import yaml

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR   = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, ROOT_DIR)

from utils.rna_utils import gc_content, extract_structural_features

log = logging.getLogger(__name__)

BASE_TO_IDX = {"A": 0, "U": 1, "G": 2, "C": 3}

# Structural feature names in a fixed order (for reproducibility)
STRUCT_FEAT_NAMES = ["gc_content", "mfe", "n_stems", "n_pairs",
                     "n_loops", "stem_fraction", "loop_fraction"]


# ─── Dataset ──────────────────────────────────────────────────────────────────

class AptamerDataset(Dataset):
    """
    Dataset pairing:
      • one-hot sequence tensor  (4, max_len)
      • structural feature vector (n_struct_features,)
      • target score (scalar, normalised to [0,1])
    """

    def __init__(
        self,
        sequences:    List[str],
        struct_feats: np.ndarray,   # shape (N, n_feats)
        scores:       List[float],
        max_len:      int = 60,
    ):
        self.max_len = max_len
        self.X_seq   = torch.stack([self._encode(s) for s in sequences])
        self.X_feat  = torch.tensor(struct_feats, dtype=torch.float32)
        self.y       = torch.tensor(scores,        dtype=torch.float32)

    def _encode(self, seq: str) -> torch.Tensor:
        t = torch.zeros(4, self.max_len)
        for i, base in enumerate(seq[:self.max_len]):
            idx = BASE_TO_IDX.get(base.upper(), 0)
            t[idx, i] = 1.0
        return t

    def __len__(self)  -> int:
        return len(self.y)

    def __getitem__(self, idx):
        return self.X_seq[idx], self.X_feat[idx], self.y[idx]


# ─── Model ────────────────────────────────────────────────────────────────────

class SequenceSelfAttention(nn.Module):
    """
    Thin wrapper: applies nn.MultiheadAttention over the time dimension of a
    CNN feature map and returns a fixed-size global summary vector.

    Input:  (batch, channels, time)  — the CNN output
    Output: (batch, channels)        — attention-pooled representation

    This is standard PyTorch; no external attention libraries are used.
    """

    def __init__(self, channels: int, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        # MultiheadAttention expects (time, batch, channels)
        self.attn = nn.MultiheadAttention(
            embed_dim   = channels,
            num_heads   = n_heads,
            dropout     = dropout,
            batch_first = False,    # PyTorch ≥1.9 supports batch_first=True too
        )
        self.pool = nn.AdaptiveAvgPool1d(1)   # global average over attended features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, C, T) → transpose to (T, batch, C) for MultiheadAttention
        x_t = x.permute(2, 0, 1)
        attn_out, _ = self.attn(x_t, x_t, x_t)   # self-attention
        # attn_out: (T, batch, C) → (batch, C, T)
        attn_out = attn_out.permute(1, 2, 0)
        # Global average pool over time → (batch, C)
        return self.pool(attn_out).squeeze(-1)


class AptamerCNN(nn.Module):
    """
    1D CNN + self-attention + structural features → binding score.

    Architecture
    ─────────────
    Sequence branch:
      Conv(4, hidden, k=5) → ReLU → Conv(hidden, hidden*2, k=5) → ReLU
      → parallel branches:
          a) global-max-pool  → (hidden*2,)
          b) self-attention   → (hidden*2,)   [if use_attention]
    Feature branch:
      Structural features → passed directly to MLP

    MLP head:
      [max-pooled ‖ attn-pooled ‖ struct_feats] → Linear(hidden) → ReLU
      → Dropout → Linear(1)
    """

    def __init__(
        self,
        max_len:      int  = 60,
        hidden:       int  = 64,
        use_attention:bool = True,
        n_struct_feats:int = 7,
        n_heads:      int  = 4,
        dropout:      float= 0.3,
    ):
        super().__init__()
        self.use_attention = use_attention

        self.conv = nn.Sequential(
            nn.Conv1d(4,      hidden,    kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(hidden, hidden*2,  kernel_size=5, padding=2),
            nn.ReLU(),
        )
        self.global_max_pool = nn.AdaptiveMaxPool1d(1)

        if use_attention:
            self.attn_head = SequenceSelfAttention(hidden*2, n_heads, dropout=0.1)
            seq_dim = hidden * 4   # max-pool (h*2) + attn-pool (h*2)
        else:
            seq_dim = hidden * 2

        self.fc = nn.Sequential(
            nn.Linear(seq_dim + n_struct_feats, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(
        self,
        x_seq:  torch.Tensor,             # (batch, 4, max_len)
        x_feat: Optional[torch.Tensor],   # (batch, n_struct_feats) or None
    ) -> torch.Tensor:
        h = self.conv(x_seq)                              # (batch, hidden*2, L)
        mp = self.global_max_pool(h).squeeze(-1)          # (batch, hidden*2)

        if self.use_attention:
            ap = self.attn_head(h)                        # (batch, hidden*2)
            seq_repr = torch.cat([mp, ap], dim=1)         # (batch, hidden*4)
        else:
            seq_repr = mp

        if x_feat is not None and x_feat.shape[-1] > 0:
            combined = torch.cat([seq_repr, x_feat], dim=1)
        else:
            combined = seq_repr

        return self.fc(combined).squeeze(-1)


# ─── Data loading ─────────────────────────────────────────────────────────────

def load_all_data(result_dir: str) -> pd.DataFrame:
    """
    Load ranked CSVs + detailed score CSVs from all rounds.
    Merges structural features (mfe, structure) from detailed files.
    Deduplicates by sequence (keep highest-scored version).
    """
    ranked_files   = sorted(Path(result_dir).glob("round_*_ranked.csv"))
    detailed_files = sorted(Path(result_dir).glob("round_*_scores_detailed.csv"))

    if not ranked_files:
        return pd.DataFrame()

    rank_dfs   = [pd.read_csv(f) for f in ranked_files]
    rank_df    = pd.concat(rank_dfs, ignore_index=True)

    # Merge structural features from detailed files if available
    if detailed_files:
        det_dfs  = [pd.read_csv(f) for f in detailed_files]
        det_df   = pd.concat(det_dfs, ignore_index=True)

        # Keep one row per seq_id (first occurrence has the structure info)
        struct_cols = ["seq_id", "structure", "mfe", "gc_content"]
        struct_cols = [c for c in struct_cols if c in det_df.columns]
        det_struct  = det_df[struct_cols].drop_duplicates("seq_id")
        rank_df     = rank_df.merge(det_struct, on="seq_id", how="left",
                                    suffixes=("", "_det"))
        # Prefer detailed version of mfe/gc_content if both present
        for col in ["mfe", "gc_content"]:
            if f"{col}_det" in rank_df.columns:
                rank_df[col] = rank_df[col].fillna(rank_df[f"{col}_det"])
                rank_df.drop(columns=[f"{col}_det"], inplace=True)

    # Compute GC if missing
    mask = rank_df["gc_content"].isna() & rank_df["sequence"].notna()
    rank_df.loc[mask, "gc_content"] = rank_df.loc[mask, "sequence"].apply(gc_content)

    rank_df = rank_df.dropna(subset=["sequence", "mean_score"])
    # Deduplicate: keep best-scoring version of each unique sequence
    rank_df = (rank_df
               .sort_values("mean_score", ascending=False)
               .drop_duplicates("sequence")
               .reset_index(drop=True))
    return rank_df


def build_structural_features(df: pd.DataFrame) -> np.ndarray:
    """
    Build structural feature matrix (N × len(STRUCT_FEAT_NAMES)).
    Uses precomputed mfe/gc_content where available; calls ViennaRNA fallback
    only if structure string is present.
    """
    feats = []
    for _, row in df.iterrows():
        seq       = row.get("sequence", "")
        structure = row.get("structure", None)
        mfe_val   = float(row.get("mfe", 0.0) or 0.0)
        gc_val    = float(row.get("gc_content", gc_content(seq)) or 0.0)

        if pd.notna(structure) and isinstance(structure, str) and len(structure) == len(seq):
            sf = extract_structural_features(structure, mfe_val)
        else:
            # Structural features unavailable — use defaults
            sf = {"n_stems": 0.0, "n_pairs": 0.0, "n_loops": 0.0,
                  "stem_fraction": 0.0, "loop_fraction": 1.0, "mfe": mfe_val}
        sf["gc_content"] = gc_val

        feats.append([sf.get(k, 0.0) for k in STRUCT_FEAT_NAMES])

    return np.array(feats, dtype=np.float32)


def normalize_features(
    train_feats: np.ndarray,
    val_feats:   Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray, np.ndarray]:
    """Min-max normalise structural features using training-set statistics."""
    lo = train_feats.min(axis=0)
    hi = train_feats.max(axis=0)
    rng = hi - lo
    rng[rng < 1e-9] = 1.0   # avoid division by zero for constant features

    train_norm = (train_feats - lo) / rng
    val_norm   = ((val_feats - lo) / rng) if val_feats is not None else None
    return train_norm, val_norm, lo, hi


# ─── Training loop ────────────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total = 0.0
    for X_seq, X_feat, y in loader:
        X_seq = X_seq.to(device)
        X_feat = X_feat.to(device)
        y      = y.to(device)
        optimizer.zero_grad()
        pred = model(X_seq, X_feat)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()
        total += loss.item() * len(y)
    return total / len(loader.dataset)


def eval_epoch(model, loader, criterion, device):
    model.eval()
    total = 0.0
    preds, trues = [], []
    with torch.no_grad():
        for X_seq, X_feat, y in loader:
            X_seq = X_seq.to(device)
            X_feat = X_feat.to(device)
            y      = y.to(device)
            pred   = model(X_seq, X_feat)
            total += criterion(pred, y).item() * len(y)
            preds.extend(pred.cpu().numpy())
            trues.extend(y.cpu().numpy())
    mae = float(np.mean(np.abs(np.array(preds) - np.array(trues))))
    return total / len(loader.dataset), mae


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train aptamer binding-score CNN+attention")
    parser.add_argument("--config",   default=os.path.join(ROOT_DIR, "config", "config.yaml"))
    parser.add_argument("--data-dir", default=os.path.join(ROOT_DIR, "results", "rankings"))
    parser.add_argument("--out-dir",  default=os.path.join(ROOT_DIR, "models"))
    args = parser.parse_args()

    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(os.path.dirname(os.path.join(ROOT_DIR, cfg["logging"]["file"])), exist_ok=True)

    logging.basicConfig(
        level=getattr(logging, cfg["logging"]["level"], logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(os.path.join(ROOT_DIR, cfg["logging"]["file"]), mode="a"),
        ],
    )

    ml     = cfg["ml"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Training device: {device}")

    df = load_all_data(args.data_dir)
    if df.empty or len(df) < 10:
        log.error(f"Not enough training data ({len(df)} samples; need ≥10). "
                  f"Run SELEX rounds first.")
        sys.exit(1)
    log.info(f"Training on {len(df)} aptamers")

    seqs   = df["sequence"].tolist()
    scores = df["mean_score"].values.astype(np.float32)

    # Normalise scores to [0,1]
    s_min, s_max = scores.min(), scores.max()
    if s_max - s_min < 1e-6:
        log.warning("All scores identical — model cannot learn. Exiting.")
        sys.exit(0)
    scores_norm = (scores - s_min) / (s_max - s_min)

    max_len = min(max(len(s) for s in seqs), 100)

    # Build structural features
    use_struct = bool(ml.get("use_structural_features", True))
    if use_struct:
        struct_feats = build_structural_features(df)
        log.info(f"Structural features: {struct_feats.shape[1]} per aptamer")
    else:
        struct_feats = np.zeros((len(seqs), 0), dtype=np.float32)

    # Train / val split
    n    = len(seqs)
    n_tr = max(1, int(n * ml["train_split"]))
    rng  = np.random.default_rng(cfg["selex"]["random_seed"])
    idx  = rng.permutation(n)
    tr_i, va_i = idx[:n_tr], idx[n_tr:]

    tr_struct, va_struct, feat_lo, feat_hi = normalize_features(
        struct_feats[tr_i],
        struct_feats[va_i] if len(va_i) > 0 else None,
    )

    train_ds = AptamerDataset(
        [seqs[i] for i in tr_i], tr_struct, [float(scores_norm[i]) for i in tr_i], max_len)
    val_ds = (AptamerDataset(
        [seqs[i] for i in va_i], va_struct, [float(scores_norm[i]) for i in va_i], max_len)
              if len(va_i) > 0 else None)

    train_loader = DataLoader(train_ds, batch_size=ml["batch_size"], shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=ml["batch_size"]) if val_ds else None

    model = AptamerCNN(
        max_len       = max_len,
        hidden        = ml["hidden_channels"],
        use_attention = bool(ml.get("use_attention", True)),
        n_struct_feats= struct_feats.shape[1] if use_struct else 0,
        n_heads       = int(ml.get("attention_heads", 4)),
        dropout       = float(ml.get("dropout", 0.3)),
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=ml["learning_rate"])
    criterion = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, factor=0.5, verbose=False)

    best_val_loss = float("inf")
    history       = []

    for epoch in range(1, ml["epochs"] + 1):
        tr_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        row = {"epoch": epoch, "train_loss": tr_loss, "val_loss": None, "val_mae": None}

        if val_loader:
            va_loss, va_mae = eval_epoch(model, val_loader, criterion, device)
            row["val_loss"] = va_loss
            row["val_mae"]  = va_mae
            scheduler.step(va_loss)

            if va_loss < best_val_loss:
                best_val_loss = va_loss
                torch.save(model.state_dict(),
                           os.path.join(args.out_dir, "best_model.pt"))

            if epoch % 10 == 0 or epoch == 1:
                log.info(
                    f"  Epoch {epoch:3d}/{ml['epochs']} | "
                    f"train={tr_loss:.4f} | val={va_loss:.4f} | mae={va_mae:.4f}"
                )
        else:
            scheduler.step(tr_loss)
            if epoch % 10 == 0 or epoch == 1:
                log.info(f"  Epoch {epoch:3d}/{ml['epochs']} | train={tr_loss:.4f}")

        history.append(row)

    torch.save(model.state_dict(), os.path.join(args.out_dir, "final_model.pt"))

    model_meta = {
        "max_len":              max_len,
        "hidden_channels":      ml["hidden_channels"],
        "use_attention":        bool(ml.get("use_attention", True)),
        "attention_heads":      int(ml.get("attention_heads", 4)),
        "n_structural_features":struct_feats.shape[1],
        "struct_feat_names":    STRUCT_FEAT_NAMES,
        "feat_lo":              feat_lo.tolist(),
        "feat_hi":              feat_hi.tolist(),
        "s_min":                float(s_min),
        "s_max":                float(s_max),
        "architecture":         "AptamerCNN_v2_attention",
    }
    with open(os.path.join(args.out_dir, "model_config.json"), "w") as fh:
        json.dump(model_meta, fh, indent=2)

    pd.DataFrame(history).to_csv(
        os.path.join(args.out_dir, "training_history.csv"), index=False)

    log.info(f"Model saved → {args.out_dir}  (best val_loss={best_val_loss:.4f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
