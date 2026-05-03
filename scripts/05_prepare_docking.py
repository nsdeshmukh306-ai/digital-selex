#!/usr/bin/env python3
"""
Script 05 – Prepare PDBQT files for AutoDock Vina.

For each TCA metabolite:  SDF → PDBQT (via meeko + RDKit)
For each RNA aptamer 3D:  PDB → PDBQT (via custom RNA atom-type converter)

Also computes and saves grid box parameters for each receptor.
Grid box center = geometric centroid of receptor; size = 20×20×20 Å.
"""

import argparse
import csv
import json
import logging
import os
import sys

import yaml

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR   = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)

from utils.docking_utils import (
    sdf_to_pdbqt_ligand,
    pdb_to_pdbqt_receptor,
    estimate_grid_center,
)

log = logging.getLogger(__name__)


def prepare_ligands(targets: list, ligand_dir: str, pdbqt_lig_dir: str) -> dict:
    """Convert all metabolite SDFs to PDBQT. Returns {name: pdbqt_path}."""
    ok = {}
    for t in targets:
        name     = t["name"]
        sdf_path = os.path.join(ligand_dir, f"{name}.sdf")
        out_path = os.path.join(pdbqt_lig_dir, f"{name}.pdbqt")

        if not os.path.exists(sdf_path):
            log.error(f"  Ligand SDF not found: {sdf_path} – run 01_fetch_ligands.py first")
            continue

        if sdf_to_pdbqt_ligand(sdf_path, out_path):
            log.info(f"  Ligand PDBQT: {out_path}")
            ok[name] = out_path
        else:
            log.error(f"  Failed to convert ligand {name} to PDBQT")
    return ok


def prepare_receptors(
    struct_3d_dir:  str,
    pdbqt_rec_dir:  str,
    grid_box_dir:   str,
    box_size:       float,
) -> dict:
    """Convert all receptor PDBs to PDBQT and compute grid boxes."""
    grid_info = {}
    pdb_files = [f for f in os.listdir(struct_3d_dir) if f.endswith(".pdb")]
    log.info(f"Preparing {len(pdb_files)} receptor PDBQT files")

    for pdb_name in pdb_files:
        seq_id   = pdb_name.replace(".pdb", "")
        pdb_path = os.path.join(struct_3d_dir, pdb_name)
        out_path = os.path.join(pdbqt_rec_dir, f"{seq_id}.pdbqt")

        if not pdb_to_pdbqt_receptor(pdb_path, out_path):
            log.error(f"  Failed to convert receptor {seq_id}")
            continue

        cx, cy, cz = estimate_grid_center(out_path)
        grid_info[seq_id] = {
            "pdbqt":    out_path,
            "center_x": cx,
            "center_y": cy,
            "center_z": cz,
            "size_x":   box_size,
            "size_y":   box_size,
            "size_z":   box_size,
        }

        # Save individual grid config
        grid_path = os.path.join(grid_box_dir, f"{seq_id}_grid.json")
        with open(grid_path, "w") as fh:
            json.dump(grid_info[seq_id], fh, indent=2)

    log.info(f"Prepared {len(grid_info)} receptor PDBQT files")
    return grid_info


def main():
    parser = argparse.ArgumentParser(description="Prepare PDBQT files for Vina docking")
    parser.add_argument("--config",    default=os.path.join(ROOT_DIR, "config", "config.yaml"))
    parser.add_argument("--targets",   default=os.path.join(ROOT_DIR, "config", "targets.yaml"))
    parser.add_argument("--3d-dir",    dest="dir_3d",
                        default=os.path.join(ROOT_DIR, "structures", "3d"))
    parser.add_argument("--round",     type=int, default=0)
    args = parser.parse_args()

    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)
    with open(args.targets) as fh:
        targets = yaml.safe_load(fh)["targets"]

    pdbqt_lig_dir = os.path.join(ROOT_DIR, "docking", "pdbqt", "ligands")
    pdbqt_rec_dir = os.path.join(ROOT_DIR, "docking", "pdbqt", "receptors")
    grid_box_dir  = os.path.join(ROOT_DIR, "docking", "grid_boxes")
    ligand_dir    = os.path.join(ROOT_DIR, "data",    "ligands")

    for d in [pdbqt_lig_dir, pdbqt_rec_dir, grid_box_dir]:
        os.makedirs(d, exist_ok=True)

    os.makedirs(os.path.dirname(os.path.join(ROOT_DIR, cfg["logging"]["file"])), exist_ok=True)
    logging.basicConfig(
        level=getattr(logging, cfg["logging"]["level"], logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(os.path.join(ROOT_DIR, cfg["logging"]["file"]), mode="a"),
        ],
    )

    log.info("── Preparing ligand PDBQT files ─────────────")
    prepare_ligands(targets, ligand_dir, pdbqt_lig_dir)

    log.info("── Preparing receptor PDBQT files ───────────")
    prepare_receptors(
        struct_3d_dir = args.dir_3d,
        pdbqt_rec_dir = pdbqt_rec_dir,
        grid_box_dir  = grid_box_dir,
        box_size      = float(cfg["docking"]["box_size_angstrom"]),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
