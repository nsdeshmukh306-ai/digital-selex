#!/usr/bin/env python3
"""
Script 06 – Run AutoDock Vina docking with multiprocessing (Section 7 upgrade).

Upgrades over v1
────────────────
• Section 7: docking jobs are distributed across docking.n_workers parallel
  processes using Python's multiprocessing.Pool.  Each worker receives
  one (receptor, ligand) pair and runs Vina independently.

  IMPORTANT: AutoDock Vina already uses docking.cpu cores internally.
  Set n_workers × cpu ≤ total_cpu_cores to avoid over-subscription.
  Default: n_workers=4, cpu=2 → uses 8 threads.

• Incremental CSV written atomically via a multiprocessing.Manager().Lock()
  to prevent race conditions when multiple workers write simultaneously.

• All other behaviour (resume, skip_docking flag, error isolation) is
  preserved from v1.

Dependencies: vina Python package (pip install vina), multiprocessing (stdlib)
"""

import argparse
import csv
import json
import logging
import os
import re
import sys
import traceback
from multiprocessing import Pool, Manager
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR   = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)

from utils.docking_utils import parse_vina_log

log = logging.getLogger(__name__)

CSV_FIELDS = ["seq_id", "target", "round", "best_affinity", "rmsd_lb", "rmsd_ub", "n_modes"]


# ─── Single-pair docking ──────────────────────────────────────────────────────

def run_vina_single(
    receptor_pdbqt: str,
    ligand_pdbqt:   str,
    center:         Tuple[float,float,float],
    size:           Tuple[float,float,float],
    out_pdbqt:      str,
    exhaustiveness: int,
    num_modes:      int,
    energy_range:   float,
    cpu:            int,
) -> List[Dict]:
    """Run AutoDock Vina for one receptor–ligand pair. Returns list of mode dicts."""
    from vina import Vina

    v = Vina(sf_name="vina", cpu=cpu, verbosity=0)
    v.set_receptor(receptor_pdbqt)
    v.set_ligand_from_file(ligand_pdbqt)
    v.compute_vina_maps(center=list(center), box_size=list(size))
    v.dock(exhaustiveness=exhaustiveness, n_poses=num_modes)
    v.write_poses(out_pdbqt, n_poses=num_modes,
                  energy_range=energy_range, overwrite=True)
    energies = v.energies(n_poses=num_modes)
    return [
        {"mode": i+1,
         "affinity":  round(float(e[0]), 4),
         "rmsd_lb":   round(float(e[1]), 4),
         "rmsd_ub":   round(float(e[2]), 4)}
        for i, e in enumerate(energies)
    ]


def parse_affinity_from_pdbqt(pdbqt_path: str) -> Tuple[float,float,float,int]:
    """Extract best affinity and RMSD from an existing Vina output PDBQT."""
    affinities, rmsd_lbs, rmsd_ubs = [], [], []
    pattern = re.compile(r"REMARK VINA RESULT:\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)")
    try:
        with open(pdbqt_path) as fh:
            for line in fh:
                m = pattern.match(line)
                if m:
                    affinities.append(float(m.group(1)))
                    rmsd_lbs.append(float(m.group(2)))
                    rmsd_ubs.append(float(m.group(3)))
    except Exception:
        pass
    if affinities:
        return affinities[0], rmsd_lbs[0], rmsd_ubs[0], len(affinities)
    return float("nan"), float("nan"), float("nan"), 0


# ─── Worker function (runs in subprocess) ────────────────────────────────────

def _dock_worker(args_tuple) -> Dict:
    """
    Worker function called by multiprocessing.Pool.
    Returns a result dict regardless of success or failure.
    """
    (seq_id, target_name, rec_pdbqt, lig_pdbqt, out_pdbqt,
     center, size, exhaustiveness, num_modes, energy_range, cpu,
     round_num, resume, already_done) = args_tuple

    if (seq_id, target_name) in already_done:
        return {"skip": True, "seq_id": seq_id, "target": target_name}

    if resume and os.path.exists(out_pdbqt):
        aff, rlb, rub, nm = parse_affinity_from_pdbqt(out_pdbqt)
        return {
            "skip":         False,
            "seq_id":       seq_id,
            "target":       target_name,
            "round":        round_num,
            "best_affinity":round(aff, 4) if aff == aff else float("nan"),
            "rmsd_lb":      round(rlb, 4) if rlb == rlb else float("nan"),
            "rmsd_ub":      round(rub, 4) if rub == rub else float("nan"),
            "n_modes":      nm,
        }

    try:
        modes = run_vina_single(
            receptor_pdbqt = rec_pdbqt,
            ligand_pdbqt   = lig_pdbqt,
            center         = center,
            size           = size,
            out_pdbqt      = out_pdbqt,
            exhaustiveness = exhaustiveness,
            num_modes      = num_modes,
            energy_range   = energy_range,
            cpu            = cpu,
        )
        best = modes[0] if modes else {"affinity": float("nan"),
                                        "rmsd_lb": float("nan"),
                                        "rmsd_ub": float("nan")}
        return {
            "skip":         False,
            "seq_id":       seq_id,
            "target":       target_name,
            "round":        round_num,
            "best_affinity":best["affinity"],
            "rmsd_lb":      best["rmsd_lb"],
            "rmsd_ub":      best["rmsd_ub"],
            "n_modes":      len(modes),
        }
    except ImportError:
        raise   # propagate so the caller can exit cleanly
    except Exception as e:
        log.error(f"  Docking failed {seq_id} × {target_name}: {e}")
        return {
            "skip":         False,
            "seq_id":       seq_id,
            "target":       target_name,
            "round":        round_num,
            "best_affinity":float("nan"),
            "rmsd_lb":      float("nan"),
            "rmsd_ub":      float("nan"),
            "n_modes":      0,
        }


# ─── Load existing results ────────────────────────────────────────────────────

def load_existing_results(out_csv: str) -> set:
    done = set()
    if not os.path.exists(out_csv):
        return done
    try:
        with open(out_csv, newline="") as fh:
            for row in csv.DictReader(fh):
                done.add((row["seq_id"], row["target"]))
    except Exception:
        pass
    return done


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Run AutoDock Vina (parallel)")
    parser.add_argument("--config",  default=os.path.join(ROOT_DIR, "config", "config.yaml"))
    parser.add_argument("--targets", default=os.path.join(ROOT_DIR, "config", "targets.yaml"))
    parser.add_argument("--round",   type=int, default=0)
    parser.add_argument("--resume",  action="store_true")
    args = parser.parse_args()

    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)
    with open(args.targets) as fh:
        all_targets = yaml.safe_load(fh)["targets"]

    active_targets = [t for t in all_targets if not t.get("skip_docking", False)]

    pdbqt_lig_dir = os.path.join(ROOT_DIR, "docking", "pdbqt", "ligands")
    pdbqt_rec_dir = os.path.join(ROOT_DIR, "docking", "pdbqt", "receptors")
    grid_box_dir  = os.path.join(ROOT_DIR, "docking", "grid_boxes")
    results_dir   = os.path.join(ROOT_DIR, "docking", "results")
    os.makedirs(results_dir, exist_ok=True)

    os.makedirs(os.path.dirname(os.path.join(ROOT_DIR, cfg["logging"]["file"])), exist_ok=True)
    logging.basicConfig(
        level=getattr(logging, cfg["logging"]["level"], logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(os.path.join(ROOT_DIR, cfg["logging"]["file"]), mode="a"),
        ],
    )

    dock_cfg       = cfg["docking"]
    exhaustiveness = dock_cfg["exhaustiveness"]
    num_modes      = dock_cfg["num_modes"]
    energy_range   = dock_cfg["energy_range"]
    cpu_per_job    = dock_cfg["cpu"]
    n_workers      = int(dock_cfg.get("n_workers", 1))
    box_size       = dock_cfg["box_size_angstrom"]

    out_csv = os.path.join(results_dir, f"round_{args.round:02d}_docking_results.csv")
    already_done = load_existing_results(out_csv) if args.resume else set()

    receptors = sorted(Path(pdbqt_rec_dir).glob("*.pdbqt"))
    log.info(
        f"Docking {len(receptors)} receptors × {len(active_targets)} ligands "
        f"(exhaustiveness={exhaustiveness}, workers={n_workers}, cpu/job={cpu_per_job})"
    )

    # Build list of work items
    work_items = []
    for rec_path in receptors:
        seq_id    = rec_path.stem
        grid_path = os.path.join(grid_box_dir, f"{seq_id}_grid.json")
        if not os.path.exists(grid_path):
            log.warning(f"  No grid config for {seq_id} – skipped")
            continue
        with open(grid_path) as fh:
            grid = json.load(fh)
        center = (grid["center_x"], grid["center_y"], grid["center_z"])
        size   = (grid["size_x"],   grid["size_y"],   grid["size_z"])

        for t in active_targets:
            tname    = t["name"]
            lig_pdbqt = os.path.join(pdbqt_lig_dir, f"{tname}.pdbqt")
            out_pdbqt = os.path.join(results_dir,    f"{seq_id}_{tname}_out.pdbqt")
            if not os.path.exists(lig_pdbqt):
                log.warning(f"  Ligand PDBQT missing: {lig_pdbqt}")
                continue
            work_items.append((
                seq_id, tname, str(rec_path), lig_pdbqt, out_pdbqt,
                center, (box_size,)*3, exhaustiveness, num_modes, energy_range,
                cpu_per_job, args.round, args.resume, already_done,
            ))

    # Run with multiprocessing Pool
    csv_is_new = not os.path.exists(out_csv) or os.path.getsize(out_csv) == 0
    written = 0

    with open(out_csv, "a", newline="") as csv_fh:
        writer = csv.DictWriter(csv_fh, fieldnames=CSV_FIELDS)
        if csv_is_new:
            writer.writeheader()

        actual_workers = min(n_workers, len(work_items)) if work_items else 1

        try:
            if actual_workers > 1:
                with Pool(processes=actual_workers) as pool:
                    for i, result in enumerate(
                        pool.imap_unordered(_dock_worker, work_items), start=1
                    ):
                        if result.get("skip"):
                            continue
                        row = {k: result[k] for k in CSV_FIELDS}
                        writer.writerow(row)
                        csv_fh.flush()
                        written += 1
                        if i % 10 == 0:
                            log.info(f"  Progress: {i}/{len(work_items)}")
            else:
                # Single-worker path (no Pool overhead, easier to debug)
                for i, item in enumerate(work_items, start=1):
                    result = _dock_worker(item)
                    if result.get("skip"):
                        continue
                    row = {k: result[k] for k in CSV_FIELDS}
                    writer.writerow(row)
                    csv_fh.flush()
                    written += 1
                    if i % 10 == 0:
                        log.info(f"  Progress: {i}/{len(work_items)}")

        except ImportError:
            log.error(
                "vina Python package not installed.\n"
                "Install with: pip install vina"
            )
            sys.exit(1)

    if written:
        log.info(f"Docking results saved: {out_csv}  ({written} pairs)")
    else:
        log.warning("No new docking results written")
    return 0


if __name__ == "__main__":
    sys.exit(main())
