#!/usr/bin/env python3
"""
Script 02 – Generate initial random RNA aptamer pool.

Generates N random RNA sequences with GC content in [gc_min, gc_max],
saves them as FASTA and a CSV summary.
"""

import argparse
import csv
import logging
import os
import sys

import numpy as np
import yaml

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR   = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)

from utils.rna_utils import generate_random_rna, gc_content, write_fasta

log = logging.getLogger(__name__)


def generate_pool(
    pool_size:   int,
    len_min:     int,
    len_max:     int,
    gc_min:      float,
    gc_max:      float,
    seed:        int,
    round_num:   int = 0,
) -> dict:
    """Generate a pool of random RNA sequences."""
    rng = np.random.default_rng(seed)
    pool = {}
    attempts = 0
    max_attempts = pool_size * 100

    while len(pool) < pool_size and attempts < max_attempts:
        length = int(rng.integers(len_min, len_max + 1))
        seq = generate_random_rna(length, gc_min=gc_min, gc_max=gc_max, rng=rng)
        if seq is not None:
            seq_id = f"R{round_num:02d}_S{len(pool):05d}"
            pool[seq_id] = seq
        attempts += 1

    if len(pool) < pool_size:
        log.warning(
            f"Only generated {len(pool)}/{pool_size} sequences after {attempts} attempts. "
            f"Consider relaxing GC filters."
        )
    return pool


def main():
    parser = argparse.ArgumentParser(description="Generate random RNA aptamer pool")
    parser.add_argument("--config",    default=os.path.join(ROOT_DIR, "config", "config.yaml"))
    parser.add_argument("--outdir",    default=os.path.join(ROOT_DIR, "sequences"))
    parser.add_argument("--pool-size", type=int,   default=None, help="Override pool size")
    parser.add_argument("--round",     type=int,   default=0,    help="Round number (for labelling)")
    args = parser.parse_args()

    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)

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

    selex = cfg["selex"]
    pool_size = args.pool_size or selex["initial_pool_size"]
    log.info(f"Generating {pool_size} random RNA sequences (round {args.round})")

    pool = generate_pool(
        pool_size  = pool_size,
        len_min    = selex["seq_length_min"],
        len_max    = selex["seq_length_max"],
        gc_min     = selex["gc_content_min"],
        gc_max     = selex["gc_content_max"],
        seed       = selex["random_seed"] + args.round,
        round_num  = args.round,
    )

    # Save FASTA
    fasta_path = os.path.join(args.outdir, f"round_{args.round:02d}_pool.fasta")
    write_fasta(pool, fasta_path)
    log.info(f"Saved FASTA: {fasta_path}  ({len(pool)} sequences)")

    # Save CSV summary
    csv_path = os.path.join(args.outdir, f"round_{args.round:02d}_pool.csv")
    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["seq_id", "sequence", "length", "gc_content", "round"])
        writer.writeheader()
        for sid, seq in pool.items():
            writer.writerow({
                "seq_id":     sid,
                "sequence":   seq,
                "length":     len(seq),
                "gc_content": round(gc_content(seq), 4),
                "round":      args.round,
            })
    log.info(f"Saved CSV:   {csv_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
