#!/usr/bin/env python3
"""
Script 03 – Predict RNA secondary structures using ViennaRNA.

Uses RNA.fold() from the ViennaRNA Python bindings to predict the
minimum free energy (MFE) secondary structure for each sequence.

ViennaRNA: https://www.tbi.univie.ac.at/RNA/
Install:   conda install -c conda-forge viennarna

Outputs dot-bracket structures + MFE values to CSV and individual files.
"""

import argparse
import csv
import logging
import os
import sys

import yaml

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR   = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)

from utils.rna_utils import read_fasta

log = logging.getLogger(__name__)


def predict_structures(fasta_path: str, out_csv: str, out_dir: str) -> list:
    """
    Predict MFE structure for all sequences in a FASTA file.
    Returns list of dicts with seq_id, sequence, structure, mfe.
    """
    try:
        import RNA  # ViennaRNA Python bindings
    except ImportError:
        log.error(
            "ViennaRNA Python bindings not found.\n"
            "Install with: conda install -c conda-forge viennarna"
        )
        sys.exit(1)

    sequences = read_fasta(fasta_path)
    results = []

    for seq_id, seq in sequences.items():
        # RNA.fold returns (structure, mfe) for the MFE structure
        structure, mfe = RNA.fold(seq)
        results.append({
            "seq_id":    seq_id,
            "sequence":  seq,
            "structure": structure,
            "mfe":       round(mfe, 4),
            "length":    len(seq),
        })

        # Write individual .dbn file (dot-bracket notation)
        dbn_path = os.path.join(out_dir, f"{seq_id}.dbn")
        with open(dbn_path, "w") as fh:
            fh.write(f">{seq_id}\n{seq}\n{structure}  ({mfe:.2f})\n")

    # Write CSV summary
    with open(out_csv, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["seq_id", "sequence", "structure", "mfe", "length"])
        writer.writeheader()
        writer.writerows(results)

    log.info(f"Predicted structures for {len(results)} sequences → {out_csv}")
    return results


def main():
    parser = argparse.ArgumentParser(description="Predict RNA 2D structures with ViennaRNA")
    parser.add_argument("--config", default=os.path.join(ROOT_DIR, "config", "config.yaml"))
    parser.add_argument("--fasta",  required=True, help="Input FASTA file")
    parser.add_argument("--outdir", default=os.path.join(ROOT_DIR, "structures", "2d"))
    parser.add_argument("--round",  type=int, default=0)
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

    out_csv = os.path.join(args.outdir, f"round_{args.round:02d}_structures.csv")
    predict_structures(args.fasta, out_csv, args.outdir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
