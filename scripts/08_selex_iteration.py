#!/usr/bin/env python3
"""
Script 08 – Iterative SELEX loop orchestrator (research-grade v2).

Upgrades over v1
────────────────
Section 4.1 — Adaptive mutation rate
  Mutation rate decays linearly from early_mutation_rate (round 0) to
  late_mutation_rate (final round).  Early rounds explore sequence space
  broadly; late rounds refine promising sequences.

Section 4.2 — Lineage tracking
  Every sequence's parent ID and round of origin are recorded in
  results/lineage.csv.  This enables tracing evolutionary trajectories.

Section 4.3 — Cluster diversity tracking
  After each round the pool is clustered by Hamming distance.  Cluster sizes
  and representative sequences are saved to results/cluster_enrichment.csv.
  This quantifies convergence and motif enrichment across rounds.

Section 6.4 — ML pre-screening (optional)
  If a trained model exists, new candidate sequences (round ≥ 2) can be
  pre-screened by the CNN before docking; only the top ML-scoring fraction
  are forwarded to the expensive docking step.  Controlled by
  ml.prescreening_top_fraction in config.yaml.
  Set to 1.0 (default) to disable pre-screening.

Section 7 — Pool size 500 (configurable)
  initial_pool_size defaults to 500 in config.yaml; this script passes it
  through unchanged.

Full results are appended to results/full_history.csv each round.
"""

import argparse
import csv
import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR   = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, ROOT_DIR)

from utils.rna_utils import (
    mutate_sequence,
    generate_random_rna,
    gc_content,
    write_fasta,
    adaptive_mutation_rate,
    cluster_pool,
    cluster_enrichment_stats,
    LineageTracker,
)

log = logging.getLogger(__name__)
PYTHON = sys.executable


# ─── Subprocess helper ────────────────────────────────────────────────────────

def run_script(script: str, extra_args: List[str] = None) -> int:
    cmd = [PYTHON, os.path.join(SCRIPT_DIR, script)] + (extra_args or [])
    log.info(f"  Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=ROOT_DIR)
    return result.returncode


# ─── Pool selection ───────────────────────────────────────────────────────────

def select_top_sequences(ranked_csv: str, n: int) -> List[Tuple[str, str]]:
    """Return top-n (seq_id, sequence) tuples from a ranked CSV."""
    df = pd.read_csv(ranked_csv).head(n)
    return list(zip(df["seq_id"].tolist(), df["sequence"].tolist()))


# ─── ML pre-screening ─────────────────────────────────────────────────────────

def ml_prescreen(
    pool: Dict[str, str],
    models_dir: str,
    top_fraction: float,
    cfg: dict,
) -> Dict[str, str]:
    """
    Use the trained CNN to pre-screen pool sequences; retain only the top
    `top_fraction` by predicted score.  If no trained model is found,
    returns pool unchanged.

    This implements Section 6.4: ML is used ONLY for pre-screening.
    Sequences not selected here are discarded before docking.
    """
    model_path  = os.path.join(models_dir, "best_model.pt")
    config_path = os.path.join(models_dir, "model_config.json")

    if not (os.path.exists(model_path) and os.path.exists(config_path)):
        log.info("  ML pre-screen: no trained model found, skipping")
        return pool

    try:
        import torch
        from torch import nn

        with open(config_path) as fh:
            mc = json.load(fh)

        # Rebuild model architecture inline (avoids circular script imports)
        import torch.nn as nn

        max_len  = mc["max_len"]
        hidden   = mc["hidden_channels"]
        use_attn = mc.get("use_attention", False)
        n_extra  = mc.get("n_structural_features", 0)

        # Import the AptamerCNN class from script 09 by path
        import importlib.util, types
        spec = importlib.util.spec_from_file_location(
            "train_model", os.path.join(SCRIPT_DIR, "09_train_model.py"))
        train_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(train_mod)
        AptamerCNN = train_mod.AptamerCNN

        model = AptamerCNN(max_len, hidden, use_attn, n_extra)
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
        model.eval()

        BASE_TO_IDX = {"A": 0, "U": 1, "G": 2, "C": 3}

        def encode(seq):
            t = torch.zeros(4, max_len)
            for i, b in enumerate(seq[:max_len]):
                t[BASE_TO_IDX.get(b.upper(), 0), i] = 1.0
            return t

        ids   = list(pool.keys())
        seqs  = list(pool.values())
        X = torch.stack([encode(s) for s in seqs])
        with torch.no_grad():
            preds = model(X, None).numpy().flatten()

        n_keep  = max(1, int(len(ids) * top_fraction))
        top_idx = np.argsort(preds)[::-1][:n_keep]
        screened = {ids[i]: seqs[i] for i in top_idx}
        log.info(
            f"  ML pre-screen: kept {len(screened)}/{len(pool)} sequences "
            f"(top {top_fraction*100:.0f}%)"
        )
        return screened

    except Exception as e:
        log.warning(f"  ML pre-screen failed ({e}); using full pool")
        return pool


# ─── Next-round pool construction ─────────────────────────────────────────────

def build_next_pool(
    selected:        List[Tuple[str, str]],
    pool_size:       int,
    mutation_rate:   float,
    random_fraction: float,
    gc_min:          float,
    gc_max:          float,
    round_num:       int,
    seed:            int,
    tracker:         LineageTracker,
) -> Dict[str, str]:
    """
    Build next-round pool from:
      • mutated copies of selected sequences  (1 - random_fraction)
      • fresh random sequences                (random_fraction)

    Records every new sequence in the LineageTracker.
    """
    rng = np.random.default_rng(seed + round_num * 100)
    pool: Dict[str, str] = {}

    n_random  = max(1, int(pool_size * random_fraction))
    n_mutants = pool_size - n_random

    # Generate mutant sequences
    seq_idx = 0
    tries   = 0
    while seq_idx < n_mutants and tries < n_mutants * 50:
        parent_id, parent_seq = selected[seq_idx % len(selected)]
        mutant = mutate_sequence(parent_seq, mutation_rate=mutation_rate, rng=rng)
        if gc_min <= gc_content(mutant) <= gc_max:
            new_id = f"R{round_num:02d}_M{seq_idx:05d}"
            pool[new_id] = mutant
            tracker.record(new_id, round_num, parent_id=parent_id, sequence=mutant)
            seq_idx += 1
        tries += 1

    # Generate fresh random sequences
    rand_idx = 0
    tries    = 0
    while rand_idx < n_random and tries < n_random * 200:
        length = int(rng.integers(30, 51))
        seq    = generate_random_rna(length, gc_min=gc_min, gc_max=gc_max, rng=rng)
        if seq is not None:
            new_id = f"R{round_num:02d}_N{rand_idx:05d}"
            pool[new_id] = seq
            tracker.record(new_id, round_num, parent_id=None, sequence=seq)
            rand_idx += 1
        tries += 1

    log.info(
        f"  Built pool: {len(pool)} sequences "
        f"({seq_idx} mutants + {rand_idx} random), μ={mutation_rate:.3f}"
    )
    return pool


# ─── One full SELEX round ─────────────────────────────────────────────────────

def run_round(round_num: int, cfg: dict, seq_csv: str) -> str:
    """
    Run one SELEX round: 2D structure → 3D structure → docking → scoring.
    Returns path to ranked CSV or empty string on failure.
    """
    seq_dir     = os.path.join(ROOT_DIR, "sequences")
    struct_dir  = os.path.join(ROOT_DIR, "structures", "2d")
    struct3_dir = os.path.join(ROOT_DIR, "structures", "3d")
    result_dir  = os.path.join(ROOT_DIR, "results",    "rankings")
    dock_dir    = os.path.join(ROOT_DIR, "docking",    "results")

    fasta = os.path.join(seq_dir, f"round_{round_num:02d}_pool.fasta")

    # Step 3: 2D structure prediction
    rc = run_script("03_predict_2d_structure.py",
                    ["--fasta", fasta, "--outdir", struct_dir, "--round", str(round_num)])
    if rc != 0:
        log.error(f"Round {round_num}: 2D structure prediction failed")
        return ""

    struct_csv = os.path.join(struct_dir, f"round_{round_num:02d}_structures.csv")

    # Step 4: 3D structure generation (uses new structure/rna_3d.py)
    rc = run_script("04_generate_3d_structure.py",
                    ["--struct-csv", struct_csv, "--outdir", struct3_dir])
    if rc != 0:
        log.error(f"Round {round_num}: 3D structure generation failed")
        return ""

    # Step 5: Prepare PDBQT
    rc = run_script("05_prepare_docking.py",
                    ["--3d-dir", struct3_dir, "--round", str(round_num)])
    if rc != 0:
        log.error(f"Round {round_num}: PDBQT preparation failed")
        return ""

    # Step 6: Run docking (Section 7 multiprocessing is inside 06_run_docking.py)
    rc = run_script("06_run_docking.py", ["--round", str(round_num), "--resume"])
    if rc != 0:
        log.error(f"Round {round_num}: Docking failed")
        return ""

    docking_csv = os.path.join(dock_dir, f"round_{round_num:02d}_docking_results.csv")

    # Step 7: Score (normalized + rescoring + negative selection)
    rc = run_script("07_score_aptamers.py", [
        "--docking-csv", docking_csv,
        "--struct-csv",  struct_csv,
        "--seq-csv",     seq_csv,
        "--outdir",      result_dir,
        "--round",       str(round_num),
    ])
    if rc != 0:
        log.error(f"Round {round_num}: Scoring failed")
        return ""

    return os.path.join(result_dir, f"round_{round_num:02d}_ranked.csv")


# ─── Full history accumulator ─────────────────────────────────────────────────

def append_to_full_history(ranked_csv: str, round_num: int, out_path: str) -> None:
    """Append this round's ranked data to results/full_history.csv."""
    if not os.path.exists(ranked_csv):
        return
    df = pd.read_csv(ranked_csv)
    df["round"] = round_num
    write_header = not os.path.exists(out_path)
    df.to_csv(out_path, mode="a", index=False, header=write_header)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Iterative SELEX loop v2")
    parser.add_argument("--config",        default=os.path.join(ROOT_DIR, "config", "config.yaml"))
    parser.add_argument("--targets",       default=os.path.join(ROOT_DIR, "config", "targets.yaml"))
    parser.add_argument("--rounds",        type=int, default=None)
    parser.add_argument("--start-round",   type=int, default=0)
    args = parser.parse_args()

    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)

    os.makedirs(os.path.dirname(os.path.join(ROOT_DIR, cfg["logging"]["file"])), exist_ok=True)
    logging.basicConfig(
        level=getattr(logging, cfg["logging"]["level"], logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(os.path.join(ROOT_DIR, cfg["logging"]["file"]), mode="a"),
        ],
    )

    selex      = cfg["selex"]
    n_rounds   = args.rounds or selex["rounds"]
    pool_size  = selex["initial_pool_size"]
    sel_frac   = selex["selection_fraction"]
    rand_frac  = selex["random_fraction"]
    gc_min     = selex["gc_content_min"]
    gc_max     = selex["gc_content_max"]
    seed       = selex["random_seed"]
    early_mu   = selex.get("early_mutation_rate", 0.15)
    late_mu    = selex.get("late_mutation_rate",  0.05)
    track_lin  = selex.get("track_lineage", True)
    cluster    = selex.get("cluster_sequences", True)
    clust_thr  = selex.get("cluster_threshold", 0.20)

    ml_cfg           = cfg.get("ml", {})
    prescreening_frac= float(ml_cfg.get("prescreening_top_fraction", 1.0))

    results_dir     = os.path.join(ROOT_DIR, "results")
    result_rank_dir = os.path.join(results_dir, "rankings")
    lineage_path    = os.path.join(results_dir, "lineage.csv")
    cluster_path    = os.path.join(results_dir, "cluster_enrichment.csv")
    history_path    = os.path.join(results_dir, "full_history.csv")
    seq_dir         = os.path.join(ROOT_DIR, "sequences")
    models_dir      = os.path.join(ROOT_DIR, "models")

    os.makedirs(result_rank_dir, exist_ok=True)
    os.makedirs(seq_dir, exist_ok=True)

    # Initialise lineage tracker
    tracker = LineageTracker()
    if track_lin and args.start_round > 0 and os.path.exists(lineage_path):
        tracker.load(lineage_path)

    cluster_history: List[Dict] = []

    log.info(
        f"═══ Digital SELEX v2: {n_rounds} rounds, "
        f"pool={pool_size}, μ_early={early_mu}, μ_late={late_mu} ═══"
    )

    start_round = args.start_round

    if start_round == 0:
        # ── Round 0: generate initial random pool ────────────────────────────
        log.info("Round 0: Generating initial random pool …")
        rc = run_script("02_generate_sequences.py", ["--round", "0"])
        if rc != 0:
            log.error("Initial pool generation failed"); sys.exit(1)

        # Register initial pool in lineage tracker
        if track_lin:
            seq_csv_0 = os.path.join(seq_dir, "round_00_pool.csv")
            if os.path.exists(seq_csv_0):
                import pandas as _pd
                init_df = _pd.read_csv(seq_csv_0)
                for _, row in init_df.iterrows():
                    tracker.record(row["seq_id"], 0, parent_id=None,
                                   sequence=row["sequence"])

        seq_csv_0  = os.path.join(seq_dir, "round_00_pool.csv")
        ranked_csv = run_round(0, cfg, seq_csv_0)
        if not ranked_csv:
            log.error("Round 0 failed"); sys.exit(1)

        append_to_full_history(ranked_csv, 0, history_path)

        # Cluster initial pool
        if cluster:
            from utils.rna_utils import read_fasta as _rfasta
            fasta0 = os.path.join(seq_dir, "round_00_pool.fasta")
            if os.path.exists(fasta0):
                pool0 = _rfasta(fasta0)
                cl0   = cluster_pool(pool0, threshold=clust_thr)
                cluster_history.append(cl0)
                log.info(f"  Round 0: {len(cl0)} clusters from {len(pool0)} sequences")

    else:
        ranked_csv = os.path.join(result_rank_dir, f"round_{start_round-1:02d}_ranked.csv")
        if not os.path.exists(ranked_csv):
            log.error(f"Cannot resume: {ranked_csv} not found"); sys.exit(1)
        log.info(f"Resuming from existing round {start_round-1} rankings")

    # ── Subsequent rounds ─────────────────────────────────────────────────────
    for rnd in range(max(1, start_round), n_rounds):
        log.info(f"\n═══ SELEX Round {rnd}/{n_rounds-1} ═══")

        # Adaptive mutation rate
        mu = adaptive_mutation_rate(rnd, n_rounds, early_mu, late_mu)
        log.info(f"  Adaptive mutation rate: {mu:.4f}")

        n_select = max(1, int(pool_size * sel_frac))
        selected = select_top_sequences(ranked_csv, n_select)
        if not selected:
            log.error("No aptamers survived scoring — check docking results")
            sys.exit(1)

        # Build next pool with lineage tracking
        next_pool = build_next_pool(
            selected        = selected,
            pool_size       = pool_size,
            mutation_rate   = mu,
            random_fraction = rand_frac,
            gc_min          = gc_min,
            gc_max          = gc_max,
            round_num       = rnd,
            seed            = seed,
            tracker         = tracker,
        )

        # Optional ML pre-screening (Section 6.4)
        if prescreening_frac < 1.0:
            next_pool = ml_prescreen(next_pool, models_dir, prescreening_frac, cfg)

        # Cluster for diversity tracking (Section 4.3)
        if cluster:
            cl = cluster_pool(next_pool, threshold=clust_thr)
            cluster_history.append(cl)
            log.info(f"  Clustering: {len(cl)} clusters from {len(next_pool)} sequences")

        # Save pool as FASTA + CSV
        fasta_path   = os.path.join(seq_dir, f"round_{rnd:02d}_pool.fasta")
        seq_csv_path = os.path.join(seq_dir, f"round_{rnd:02d}_pool.csv")
        write_fasta(next_pool, fasta_path)

        with open(seq_csv_path, "w", newline="") as fh:
            writer = csv.DictWriter(
                fh, fieldnames=["seq_id","sequence","length","gc_content","round"])
            writer.writeheader()
            for sid, seq in next_pool.items():
                writer.writerow({
                    "seq_id":     sid,
                    "sequence":   seq,
                    "length":     len(seq),
                    "gc_content": round(gc_content(seq), 4),
                    "round":      rnd,
                })

        ranked_csv = run_round(rnd, cfg, seq_csv_path)
        if not ranked_csv:
            log.error(f"Round {rnd} failed; stopping early")
            break

        append_to_full_history(ranked_csv, rnd, history_path)

    # ── Save lineage + cluster enrichment ────────────────────────────────────
    if track_lin and len(tracker) > 0:
        tracker.save(lineage_path)

    if cluster and cluster_history:
        stats = cluster_enrichment_stats(cluster_history)
        os.makedirs(results_dir, exist_ok=True)
        if stats:
            cl_df = pd.DataFrame(stats)
            cl_df.to_csv(cluster_path, index=False)
            log.info(f"Cluster enrichment saved: {cluster_path}")

    log.info(f"\n═══ SELEX complete. Final rankings: {ranked_csv} ═══")
    log.info(f"    Full history: {history_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
