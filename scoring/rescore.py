"""
scoring/rescore.py — Geometric rescoring of docked poses.

Rationale
---------
AutoDock Vina's empirical scoring function was parameterised on protein–
ligand complexes.  For RNA–small-molecule interactions, the score provides
a useful relative ranking but can systematically under- or over-estimate
contributions from:
  • RNA 2'-OH hydrogen bonds (not parametrised in Vina force field)
  • Electrostatic phosphate–ligand repulsion (partially screened by ions)
  • Base stacking of aromatic ligands

This module adds a fast geometric rescoring layer computed directly from
the atom coordinates in the Vina output PDBQT file (receptor) and the
docked ligand pose (also PDBQT).

Components implemented
----------------------
1. H-bond count score
   Counts (donor, acceptor) pairs within hbond_dist_cutoff (default 3.5 Å).
   H-bond donors: N, O atoms that carry implied H's (atom types NA, OA, N, OD).
   H-bond acceptors: N, O atoms (atom types NA, OA, OD, N).
   APPROXIMATION: explicit hydrogens are not present in the PDBQT (meeko
   removes them for docking); donor/acceptor assignment is based on
   AutoDock atom types only.  Geometric angle criterion is NOT applied
   (would need H positions).  This gives a rough count, not a calibrated
   energy term.

2. Close-contact count score
   Counts heavy-atom pairs across receptor–ligand interface within
   contact_dist_cutoff (default 4.0 Å).  This is a proxy for the buried
   surface area and general complementarity.

3. Approximate interaction heuristic
   A simple pairwise ε/r^6 (van der Waals attractive) sum over all
   receptor–ligand heavy-atom pairs.  This is NOT a calibrated force-field
   term; it is a relative ranking signal only.

Normalisation
-------------
Each component is normalised to [0, 1] across the pool of docked poses
seen so far, using min-max scaling with a small epsilon for stability.

All weights are loaded from config.yaml (scoring.hbond_weight,
scoring.contact_weight).

Dependencies
------------
numpy  (standard)
"""

import logging
import os
import re
from typing import Dict, List, Optional, Tuple

import numpy as np

log = logging.getLogger(__name__)

# AutoDock atom types classified as H-bond donors or acceptors.
# Source: AutoDock4 atom type definitions.
# NA = N acceptor, OA = O acceptor, N = N (donor/acceptor), OD = O donor
_HBOND_DONOR_TYPES    = {"NA", "N", "OD"}   # can donate H
_HBOND_ACCEPTOR_TYPES = {"NA", "OA", "OD", "N"}  # can accept H


def _parse_pdbqt_atoms(pdbqt_path: str) -> Tuple[np.ndarray, List[str]]:
    """
    Parse ATOM/HETATM lines from a PDBQT file.
    Returns:
        coords : (N, 3) float array of (x, y, z)
        types  : list of AutoDock atom type strings (length N)
    """
    coords = []
    types  = []
    try:
        with open(pdbqt_path) as fh:
            for line in fh:
                if not line.startswith(("ATOM", "HETATM")):
                    continue
                try:
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                    # AutoDock type is in cols 78-79 (last field)
                    ad_type = line[77:79].strip() if len(line) > 77 else "C"
                    coords.append((x, y, z))
                    types.append(ad_type)
                except (ValueError, IndexError):
                    continue
    except FileNotFoundError:
        log.warning(f"PDBQT not found: {pdbqt_path}")
    return np.array(coords, dtype=float), types


def _parse_pdbqt_best_pose(pdbqt_path: str) -> Tuple[np.ndarray, List[str]]:
    """
    Parse the FIRST docking pose (MODEL 1) from a Vina output PDBQT.
    Returns (coords, types) for the best-scoring pose.
    """
    coords = []
    types  = []
    in_model = False
    try:
        with open(pdbqt_path) as fh:
            for line in fh:
                if line.startswith("MODEL"):
                    in_model = True
                    continue
                if line.startswith("ENDMDL") and in_model:
                    break
                if in_model and line.startswith(("ATOM", "HETATM")):
                    try:
                        x = float(line[30:38])
                        y = float(line[38:46])
                        z = float(line[46:54])
                        ad_type = line[77:79].strip() if len(line) > 77 else "C"
                        coords.append((x, y, z))
                        types.append(ad_type)
                    except (ValueError, IndexError):
                        continue
    except FileNotFoundError:
        log.warning(f"Pose PDBQT not found: {pdbqt_path}")
    # Fallback if no MODEL marker: parse all atoms
    if not coords:
        return _parse_pdbqt_atoms(pdbqt_path)
    return np.array(coords, dtype=float), types


def count_hbonds(
    rec_coords: np.ndarray,
    rec_types:  List[str],
    lig_coords: np.ndarray,
    lig_types:  List[str],
    cutoff:     float = 3.5,
) -> int:
    """
    Count potential H-bonds between receptor and ligand atoms.

    A putative H-bond is counted when a donor-type atom in one molecule
    is within `cutoff` Å of an acceptor-type atom in the other molecule.
    Angular criterion is omitted because H positions are not available.

    APPROXIMATION: counts may be inflated by parallel dipoles that are
    not true H-bonds.  Use only for relative ranking within a docked pool.
    """
    if len(rec_coords) == 0 or len(lig_coords) == 0:
        return 0

    # Pairwise distance matrix via broadcasting
    # Shape: (n_rec, n_lig)
    diff  = rec_coords[:, np.newaxis, :] - lig_coords[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=-1)

    hbond_count = 0
    for i, (rt, row) in enumerate(zip(rec_types, dists)):
        for j, (lt, d) in enumerate(zip(lig_types, row)):
            if d > cutoff:
                continue
            # Donor in receptor, acceptor in ligand, or vice versa
            if (rt in _HBOND_DONOR_TYPES and lt in _HBOND_ACCEPTOR_TYPES) or \
               (rt in _HBOND_ACCEPTOR_TYPES and lt in _HBOND_DONOR_TYPES):
                hbond_count += 1

    return hbond_count


def count_contacts(
    rec_coords: np.ndarray,
    lig_coords: np.ndarray,
    cutoff:     float = 4.0,
) -> int:
    """
    Count heavy-atom contacts between receptor and ligand within `cutoff` Å.
    All atom-type pairs are counted (non-selective).
    This is a proxy for buried interface area.
    """
    if len(rec_coords) == 0 or len(lig_coords) == 0:
        return 0
    diff  = rec_coords[:, np.newaxis, :] - lig_coords[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=-1)
    return int((dists < cutoff).sum())


def vdw_attractive_sum(
    rec_coords: np.ndarray,
    lig_coords: np.ndarray,
    sigma: float = 4.0,
    epsilon: float = 0.1,
) -> float:
    """
    Compute sum of Lennard-Jones attractive (r^-6) terms across all pairs.

    APPROXIMATION: single epsilon/sigma for all atom pairs; no repulsive term
    included (pairs < sigma are excluded).  This is purely a relative heuristic
    — NOT a calibrated force-field energy.

    Using only the attractive r^-6 branch prevents explosion from steric
    clashes and gives a smooth signal correlating with interface complementarity.
    """
    if len(rec_coords) == 0 or len(lig_coords) == 0:
        return 0.0
    diff  = rec_coords[:, np.newaxis, :] - lig_coords[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=-1)
    # Only count pairs in the [sigma, 2*sigma] attractive well
    mask  = (dists >= sigma) & (dists < 2 * sigma)
    r6    = np.where(mask, (sigma / np.maximum(dists, 1e-3)) ** 6, 0.0)
    return float(epsilon * r6.sum())


def rescore_pose(
    receptor_pdbqt: str,
    pose_pdbqt:     str,
    hbond_cutoff:   float = 3.5,
    contact_cutoff: float = 4.0,
) -> Dict[str, float]:
    """
    Compute all geometric rescoring components for one receptor–pose pair.

    Returns dict with keys:
      n_hbonds      — raw count of putative H-bonds
      n_contacts    — raw count of close contacts (< contact_cutoff Å)
      vdw_attr      — van der Waals attractive heuristic sum
    """
    rec_c, rec_t = _parse_pdbqt_atoms(receptor_pdbqt)
    lig_c, lig_t = _parse_pdbqt_best_pose(pose_pdbqt)

    return {
        "n_hbonds":   count_hbonds(rec_c, rec_t, lig_c, lig_t, hbond_cutoff),
        "n_contacts": count_contacts(rec_c, lig_c, contact_cutoff),
        "vdw_attr":   vdw_attractive_sum(rec_c, lig_c),
    }


def normalize_column(values: np.ndarray) -> np.ndarray:
    """
    Min-max normalise a 1D array to [0, 1].
    Returns array of same length; all-identical input → all zeros.
    """
    lo, hi = values.min(), values.max()
    if hi - lo < 1e-9:
        return np.zeros_like(values, dtype=float)
    return (values - lo) / (hi - lo)


def batch_rescore(
    records: List[Dict],
    pdbqt_rec_dir: str,
    docking_results_dir: str,
    hbond_cutoff:   float = 3.5,
    contact_cutoff: float = 4.0,
) -> List[Dict]:
    """
    Run geometric rescoring for a list of (seq_id, target) docking records.

    Parameters
    ----------
    records : list of dicts with keys 'seq_id', 'target', 'round'
    pdbqt_rec_dir : directory containing receptor .pdbqt files
    docking_results_dir : directory containing Vina output pose .pdbqt files

    Returns the same records list with added keys:
      n_hbonds, n_contacts, vdw_attr,
      hbond_score (normalised), contact_score (normalised)
    """
    raw_hbonds   = []
    raw_contacts = []

    for rec in records:
        seq_id = rec["seq_id"]
        target = rec["target"]
        rec_path  = os.path.join(pdbqt_rec_dir, f"{seq_id}.pdbqt")
        pose_path = os.path.join(docking_results_dir, f"{seq_id}_{target}_out.pdbqt")

        if os.path.exists(rec_path) and os.path.exists(pose_path):
            scores = rescore_pose(rec_path, pose_path, hbond_cutoff, contact_cutoff)
        else:
            scores = {"n_hbonds": 0, "n_contacts": 0, "vdw_attr": 0.0}

        rec.update(scores)
        raw_hbonds.append(scores["n_hbonds"])
        raw_contacts.append(scores["n_contacts"])

    # Normalise across the pool so that values are comparable across rounds
    hbonds_norm   = normalize_column(np.array(raw_hbonds, dtype=float))
    contacts_norm = normalize_column(np.array(raw_contacts, dtype=float))

    for i, rec in enumerate(records):
        rec["hbond_score"]   = float(hbonds_norm[i])
        rec["contact_score"] = float(contacts_norm[i])

    return records
