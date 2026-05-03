#!/usr/bin/env python3
"""
Script 07 – Score and rank aptamers (research-grade v2).

Scoring architecture (Section 5 refactor + Section 2 rescoring + Section 3)
─────────────────────────────────────────────────────────────────────────────

Component             Source                     Normalisation
─────────────────────────────────────────────────────────────────────────────
binding_score         Vina best_affinity × -1    min-max → [0,1]
stability_score       MFE × -1                   min-max → [0,1]
gc_score              1 - |gc - 0.5| / 0.5       already [0,1]
hbond_score           H-bond count (rescore.py)  min-max → [0,1]
contact_score         Close contacts (rescore.py) min-max → [0,1]
specificity_score     target / mean_off_target    min-max → [0,1]
─────────────────────────────────────────────────────────────────────────────

Final composite (per aptamer × per on-target):
  composite = w_aff × binding_score
            + w_mfe × stability_score
            + w_gc  × gc_score
            + w_hb  × hbond_score       (if use_rescore)
            + w_ct  × contact_score     (if use_rescore)
            + w_sp  × specificity_score  (if use_specificity)

Per-aptamer summary (across all on-targets):
  mean_score       = mean(composite over on-targets)
  best_target      = on-target with highest composite
  specificity_index = max / (mean + ε)

All weights are read from config.yaml so they are adjustable without
touching code.

Negative selection (Section 3)
──────────────────────────────
Off-target metabolites listed in config.yaml→scoring.off_targets are
used to compute a specificity penalisation term.  The off-target docking
results must exist in the same docking results directory.

If off-target docking results are missing the specificity component is
skipped with a logged warning (the rest of scoring proceeds normally).

APPROXIMATION: off-target metabolites must be pre-docked (they are NOT
automatically added to the docking step by this script — see run_pipeline.sh
for the updated flow that docks off-targets too).
"""

import argparse
import logging
import os
import sys
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import yaml

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR   = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, ROOT_DIR)

from utils.rna_utils import gc_content

log = logging.getLogger(__name__)

EPS = 1e-9


# ─── Normalisation ────────────────────────────────────────────────────────────

def minmax_normalize(series: pd.Series) -> pd.Series:
    """Min-max normalise a pandas Series to [0, 1]. All-same → zeros."""
    lo, hi = series.min(), series.max()
    if hi - lo < EPS:
        return pd.Series(np.zeros(len(series)), index=series.index)
    return (series - lo) / (hi - lo)


# ─── Scoring components ───────────────────────────────────────────────────────

def compute_binding_scores(dock_df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert Vina affinities to normalised binding scores.
    Vina affinity is negative (more negative = better).
    binding_score_raw = -affinity  (positive, higher = better)
    binding_score     = min-max normalised across the pool.
    """
    df = dock_df.copy()
    df["binding_score_raw"] = -df["best_affinity"].fillna(0.0)
    # Normalise across all (aptamer, target) rows together
    df["binding_score"] = minmax_normalize(df["binding_score_raw"])
    return df


def compute_stability_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Stability score from MFE (minimum free energy, kcal/mol).
    More negative MFE = more stable = higher score.
    stability_score = min-max normalised(-mfe).
    """
    df = df.copy()
    df["stability_score_raw"] = -df["mfe"].fillna(0.0)
    df["stability_score"] = minmax_normalize(df["stability_score_raw"])
    return df


def compute_gc_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    GC score: penalise deviation from 50% GC.
    gc_score = 1 - |gc - 0.5| / 0.5  → 1.0 at GC=0.5, 0.0 at GC=0 or 1.
    Already in [0,1]; no further normalisation needed.
    """
    df = df.copy()
    gc = df["gc_content"].fillna(0.5)
    df["gc_score"] = 1.0 - (gc - 0.5).abs() / 0.5
    return df


def compute_specificity_scores(
    df: pd.DataFrame,
    off_target_dock_df: Optional[pd.DataFrame],
) -> pd.DataFrame:
    """
    Specificity score: how much better does the aptamer bind on-targets
    vs. off-targets?

    specificity_raw = mean_on_target_binding / (mean_off_target_binding + ε)

    Higher = more specific.  Normalised to [0,1] across the pool.

    If off_target_dock_df is None, specificity_score is set to 0.5 (neutral).
    """
    df = df.copy()

    if off_target_dock_df is None or off_target_dock_df.empty:
        log.warning(
            "No off-target docking data — specificity component set to 0.5 (neutral). "
            "Add off_targets to config.yaml and dock them to enable negative selection."
        )
        df["specificity_score"] = 0.5
        return df

    # Per-aptamer mean off-target binding score (using same sign convention)
    off_agg = (
        off_target_dock_df
        .copy()
        .assign(off_binding=lambda x: -x["best_affinity"].fillna(0.0))
        .groupby("seq_id")["off_binding"]
        .mean()
        .rename("mean_off_binding")
    )

    # Per-aptamer mean on-target binding score (already computed before this call)
    on_agg = (
        df.groupby("seq_id")["binding_score_raw"]
        .mean()
        .rename("mean_on_binding")
    )

    spec_df = pd.concat([on_agg, off_agg], axis=1).fillna(0.0)
    spec_df["specificity_raw"] = (
        spec_df["mean_on_binding"] / (spec_df["mean_off_binding"] + EPS)
    )
    spec_df["specificity_score_norm"] = minmax_normalize(spec_df["specificity_raw"])

    df = df.merge(
        spec_df[["specificity_score_norm"]].reset_index(),
        on="seq_id", how="left",
    )
    df["specificity_score"] = df["specificity_score_norm"].fillna(0.5)
    return df


# ─── Rescoring integration ───────────────────────────────────────────────────

def add_rescore_columns(
    df: pd.DataFrame,
    cfg: dict,
    round_num: int,
) -> pd.DataFrame:
    """
    Add H-bond and contact scores from scoring/rescore.py.
    Works row-by-row so it is robust to missing pose files.
    """
    try:
        sys.path.insert(0, os.path.join(ROOT_DIR, "scoring"))
        from rescore import batch_rescore

        pdbqt_rec_dir    = os.path.join(ROOT_DIR, "docking", "pdbqt", "receptors")
        docking_results  = os.path.join(ROOT_DIR, "docking", "results")

        records = df[["seq_id", "target"]].copy()
        records["round"] = round_num
        records = records.to_dict("records")

        hb_cutoff = cfg["scoring"].get("hbond_dist_cutoff", 3.5)
        ct_cutoff = cfg["scoring"].get("contact_dist_cutoff", 4.0)

        records = batch_rescore(records, pdbqt_rec_dir, docking_results,
                                hbond_cutoff=hb_cutoff,
                                contact_cutoff=ct_cutoff)

        rescore_df = pd.DataFrame(records)[["seq_id","target","hbond_score","contact_score"]]
        df = df.merge(rescore_df, on=["seq_id","target"], how="left")
        df["hbond_score"]   = df["hbond_score"].fillna(0.0)
        df["contact_score"] = df["contact_score"].fillna(0.0)
        log.info("  Rescoring: H-bond and contact scores added")

    except ImportError as e:
        log.warning(f"  Rescoring skipped (import error: {e}); setting hbond/contact = 0")
        df["hbond_score"]   = 0.0
        df["contact_score"] = 0.0
    except Exception as e:
        log.warning(f"  Rescoring failed ({e}); setting hbond/contact = 0")
        df["hbond_score"]   = 0.0
        df["contact_score"] = 0.0

    return df


# ─── Full composite scoring ───────────────────────────────────────────────────

def compute_composite(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Apply weighted sum over all normalised score components."""
    sc   = cfg["scoring"]
    w_aff = float(sc.get("affinity_weight",    1.0))
    w_mfe = float(sc.get("mfe_weight",         0.1))
    w_gc  = float(sc.get("gc_penalty_weight",  0.05))
    w_hb  = float(sc.get("hbond_weight",       0.30)) if sc.get("use_rescore", True) else 0.0
    w_ct  = float(sc.get("contact_weight",     0.20)) if sc.get("use_rescore", True) else 0.0
    w_sp  = float(sc.get("specificity_weight", 0.25)) if sc.get("use_specificity", True) else 0.0

    df["composite_score"] = (
          w_aff * df["binding_score"]
        + w_mfe * df.get("stability_score", 0)
        + w_gc  * df.get("gc_score", 0)
        + w_hb  * df.get("hbond_score", 0)
        + w_ct  * df.get("contact_score", 0)
        + w_sp  * df.get("specificity_score", 0.5)
    )
    return df


# ─── Per-aptamer ranking ──────────────────────────────────────────────────────

def rank_aptamers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate per-(aptamer, target) scores to per-aptamer ranking.
    """
    best_idx     = df.groupby("seq_id")["composite_score"].idxmax()
    best_targets = df.loc[best_idx, ["seq_id", "target"]].set_index("seq_id")["target"]

    agg = (
        df.groupby("seq_id")
        .agg(
            sequence         = ("sequence",        "first"),
            gc_content       = ("gc_content",      "first"),
            mfe              = ("mfe",             "first"),
            mean_score       = ("composite_score", "mean"),
            min_score        = ("composite_score", "min"),
            max_score        = ("composite_score", "max"),
            n_targets_scored = ("target",          "count"),
            mean_binding     = ("binding_score",   "mean"),
            mean_stability   = ("stability_score", "mean"),
            mean_hbond       = ("hbond_score",     "mean"),
            mean_contact     = ("contact_score",   "mean"),
            mean_specificity = ("specificity_score","mean"),
        )
        .reset_index()
    )
    agg["best_target"]       = agg["seq_id"].map(best_targets)
    agg["specificity_index"] = agg["max_score"] / (agg["mean_score"].abs() + EPS)
    agg = agg.sort_values("mean_score", ascending=False).reset_index(drop=True)
    agg.insert(0, "rank", agg.index + 1)
    return agg


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Score and rank aptamers (v2)")
    parser.add_argument("--config",        default=os.path.join(ROOT_DIR, "config", "config.yaml"))
    parser.add_argument("--targets",       default=os.path.join(ROOT_DIR, "config", "targets.yaml"))
    parser.add_argument("--docking-csv",   required=True, help="On-target docking results CSV")
    parser.add_argument("--struct-csv",    required=True, help="2D structure CSV (MFE)")
    parser.add_argument("--seq-csv",       required=True, help="Sequence pool CSV (gc_content)")
    parser.add_argument("--outdir",        default=os.path.join(ROOT_DIR, "results", "rankings"))
    parser.add_argument("--round",         type=int, default=0)
    args = parser.parse_args()

    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)
    with open(args.targets) as fh:
        target_cfg = yaml.safe_load(fh)["targets"]

    os.makedirs(args.outdir, exist_ok=True)
    os.makedirs(os.path.dirname(os.path.join(ROOT_DIR, cfg["logging"]["file"])), exist_ok=True)

    logging.basicConfig(
        level=getattr(logging, cfg["logging"]["level"], logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(os.path.join(ROOT_DIR, cfg["logging"]["file"]), mode="a"),
        ],
    )

    log.info(f"═══ Scoring round {args.round} ═══")

    # ── Load data ─────────────────────────────────────────────────────────────
    dock_df   = pd.read_csv(args.docking_csv)
    struct_df = pd.read_csv(args.struct_csv)[["seq_id", "mfe", "structure"]]
    seq_df    = pd.read_csv(args.seq_csv)[["seq_id", "gc_content", "sequence"]]

    # Identify on-targets (targets that are NOT in off_targets list)
    off_target_names = set(cfg["scoring"].get("off_targets", []))
    on_target_names  = {t["name"] for t in target_cfg
                        if not t.get("skip_docking", False)
                        and t["name"] not in off_target_names}

    # Split docking results into on-target and off-target
    on_df  = dock_df[dock_df["target"].isin(on_target_names)].copy()
    off_df = dock_df[dock_df["target"].isin(off_target_names)].copy()

    if on_df.empty:
        log.warning("No on-target docking rows found; check targets.yaml and docking CSV")
        on_df = dock_df.copy()    # fallback: treat all as on-target

    # ── Merge structural data ─────────────────────────────────────────────────
    df = on_df.merge(struct_df, on="seq_id", how="left")
    df = df.merge(seq_df, on="seq_id", suffixes=("", "_seq"), how="left")

    # Fill gc_content from sequence string if missing
    mask = df["gc_content"].isna() & df["sequence"].notna()
    df.loc[mask, "gc_content"] = df.loc[mask, "sequence"].apply(gc_content)

    df = df.dropna(subset=["best_affinity"])

    if df.empty:
        log.error("No valid docking data after filtering — check docking results")
        return 1

    log.info(f"  Scoring {df['seq_id'].nunique()} aptamers × {df['target'].nunique()} targets")

    # ── Compute all score components ──────────────────────────────────────────
    df = compute_binding_scores(df)
    df = compute_stability_scores(df)
    df = compute_gc_scores(df)

    if cfg["scoring"].get("use_rescore", True):
        df = add_rescore_columns(df, cfg, args.round)

    if cfg["scoring"].get("use_specificity", True):
        df = compute_specificity_scores(df, off_df if not off_df.empty else None)

    df = compute_composite(df, cfg)

    # ── Save detailed (per aptamer × target) ─────────────────────────────────
    detail_cols = [
        "seq_id","target","round","sequence","gc_content","mfe","structure",
        "best_affinity","binding_score","stability_score","gc_score",
        "hbond_score","contact_score","specificity_score","composite_score",
    ]
    detail_cols = [c for c in detail_cols if c in df.columns]
    det_path = os.path.join(args.outdir, f"round_{args.round:02d}_scores_detailed.csv")
    df[detail_cols].to_csv(det_path, index=False)
    log.info(f"  Detailed scores: {det_path}")

    # ── Rank per aptamer ─────────────────────────────────────────────────────
    ranked_df = rank_aptamers(df)
    rank_path = os.path.join(args.outdir, f"round_{args.round:02d}_ranked.csv")
    ranked_df.to_csv(rank_path, index=False)
    log.info(f"  Ranked aptamers: {rank_path}")

    top = ranked_df.head(10)[["rank","seq_id","sequence","mean_score","best_target"]]
    log.info(f"\nTop 10 aptamers (round {args.round}):\n{top.to_string(index=False)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
