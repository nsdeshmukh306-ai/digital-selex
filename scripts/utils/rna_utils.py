"""
RNA utility functions for the digital SELEX pipeline.

New in v2 (research-grade upgrade)
───────────────────────────────────
• extract_structural_features()  — stem count, loop count, bulge count, MFE
• hamming_fraction()             — normalised Hamming distance (same length)
• cluster_pool()                 — greedy Hamming-based sequence clustering
• LineageTracker                 — records parent → child relationships per round
• adaptive_mutation_rate()       — linearly decaying mutation schedule

3D structure generation still includes the A-form coarse-grain builder which
is used as a fallback when RNAComposer is unavailable (see structure/rna_3d.py).
"""

import csv
import logging
import os
from typing import Dict, List, Optional, Tuple

import numpy as np

log = logging.getLogger(__name__)


# ─── Sequence Utilities ───────────────────────────────────────────────────────

def gc_content(seq: str) -> float:
    seq = seq.upper()
    return (seq.count("G") + seq.count("C")) / len(seq)


def generate_random_rna(
    length: int,
    gc_min: float = 0.40,
    gc_max: float = 0.60,
    rng: Optional[np.random.Generator] = None,
    max_tries: int = 500,
) -> Optional[str]:
    """
    Generate a random RNA sequence with GC content within [gc_min, gc_max].
    Returns None if no valid sequence found in max_tries attempts.
    """
    if rng is None:
        rng = np.random.default_rng()
    bases = np.array(["A", "U", "G", "C"])
    for _ in range(max_tries):
        seq = "".join(rng.choice(bases, size=length))
        gc = gc_content(seq)
        if gc_min <= gc <= gc_max:
            return seq
    return None


def mutate_sequence(
    seq: str,
    mutation_rate: float = 0.10,
    rng: Optional[np.random.Generator] = None,
) -> str:
    """Apply random point mutations; each position mutated with probability mutation_rate."""
    if rng is None:
        rng = np.random.default_rng()
    bases = ["A", "U", "G", "C"]
    seq_list = list(seq.upper())
    for i in range(len(seq_list)):
        if rng.random() < mutation_rate:
            alt = [b for b in bases if b != seq_list[i]]
            seq_list[i] = rng.choice(alt)
    return "".join(seq_list)


def write_fasta(sequences: Dict[str, str], path: str) -> None:
    with open(path, "w") as fh:
        for sid, seq in sequences.items():
            fh.write(f">{sid}\n{seq}\n")


def read_fasta(path: str) -> Dict[str, str]:
    seqs: Dict[str, str] = {}
    current = None
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line.startswith(">"):
                current = line[1:].split()[0]
                seqs[current] = ""
            elif current is not None:
                seqs[current] += line.upper()
    return seqs


# ─── Adaptive Mutation Rate ───────────────────────────────────────────────────

def adaptive_mutation_rate(
    round_num: int,
    total_rounds: int,
    early_rate: float = 0.15,
    late_rate: float = 0.05,
) -> float:
    """
    Return a linearly decaying mutation rate.

    Scientific rationale: early SELEX rounds benefit from high diversity
    (exploration); late rounds should refine top sequences with small
    mutations (exploitation).  This schedule mirrors adaptive crossover
    rates in evolutionary algorithms.
    """
    if total_rounds <= 1:
        return early_rate
    fraction = round_num / (total_rounds - 1)   # 0.0 → 1.0
    return early_rate + fraction * (late_rate - early_rate)


# ─── Secondary Structure Utilities ───────────────────────────────────────────

def parse_dot_bracket(structure: str) -> List[Tuple[int, int]]:
    """
    Parse dot-bracket notation → sorted list of (i, j) base-pair tuples.
    Handles standard parentheses only (no pseudoknots).
    """
    pairs: List[Tuple[int, int]] = []
    stack: List[int] = []
    for idx, ch in enumerate(structure):
        if ch == "(":
            stack.append(idx)
        elif ch == ")":
            if stack:
                j = stack.pop()
                pairs.append((j, idx))
    return sorted(pairs)


def _find_stems(pairs: List[Tuple[int, int]]) -> List[List[Tuple[int, int]]]:
    """
    Group consecutive base-pairs into stems (continuous helical runs).
    A stem is a maximal set of pairs (i,j), (i+1,j-1), (i+2,j-2), ...
    """
    if not pairs:
        return []
    stems: List[List[Tuple[int, int]]] = []
    current: List[Tuple[int, int]] = [pairs[0]]
    for k in range(1, len(pairs)):
        pi, pj = pairs[k - 1]
        ci, cj = pairs[k]
        if ci == pi + 1 and cj == pj - 1:
            current.append(pairs[k])
        else:
            stems.append(current)
            current = [pairs[k]]
    stems.append(current)
    return stems


def extract_structural_features(structure: str, mfe: float = 0.0) -> Dict[str, float]:
    """
    Extract interpretable features from a ViennaRNA dot-bracket structure.

    Features
    --------
    n_stems       : number of stem (helix) regions
    n_pairs       : total number of base pairs
    n_loops       : number of loop/bulge/hairpin regions (approximate)
    stem_fraction  : fraction of positions involved in stems
    loop_fraction  : fraction of unpaired positions
    mfe           : minimum free energy (pass-through for convenience)

    Method: stems are identified as consecutive stacked base pairs.
    Loops are counted as contiguous runs of unpaired ('.') nucleotides
    that are flanked by paired regions on at least one side.

    Scientific note: stem count and MFE are correlated with structural
    stability.  More stems → more negative MFE → more stable aptamer.
    Aptamers with mixed stem-loop topology are often better binders
    because structured loops can form tight pockets around small molecules.
    """
    n = len(structure)
    if n == 0:
        return {"n_stems": 0, "n_pairs": 0, "n_loops": 0,
                "stem_fraction": 0.0, "loop_fraction": 1.0, "mfe": mfe}

    pairs = parse_dot_bracket(structure)
    stems = _find_stems(pairs)

    paired_positions = set()
    for i, j in pairs:
        paired_positions.add(i)
        paired_positions.add(j)

    # Count contiguous unpaired runs flanked by paired positions
    in_loop     = False
    loop_count  = 0
    for idx, ch in enumerate(structure):
        is_paired = idx in paired_positions
        if not is_paired and not in_loop:
            # Start of an unpaired run — check if it's inside the structure
            # (i.e., it has paired neighbours — hairpin loop or internal loop)
            left_has_pair  = any(p in paired_positions for p in range(max(0, idx-3), idx))
            right_has_pair = any(p in paired_positions for p in range(idx+1, min(n, idx+4)))
            if left_has_pair or right_has_pair:
                loop_count += 1
            in_loop = True
        elif is_paired:
            in_loop = False

    return {
        "n_stems":       float(len(stems)),
        "n_pairs":       float(len(pairs)),
        "n_loops":       float(loop_count),
        "stem_fraction": float(len(paired_positions)) / n,
        "loop_fraction": float(n - len(paired_positions)) / n,
        "mfe":           float(mfe),
    }


# ─── Sequence Clustering (Hamming distance) ───────────────────────────────────

def hamming_fraction(seq_a: str, seq_b: str) -> float:
    """
    Normalised Hamming distance between two sequences of the same length.
    If lengths differ, the shorter sequence is zero-padded on the right.

    Returns a value in [0.0, 1.0]: 0.0 = identical, 1.0 = all different.
    """
    la, lb = len(seq_a), len(seq_b)
    max_len = max(la, lb)
    if max_len == 0:
        return 0.0
    # Pad shorter sequence
    a = seq_a.upper().ljust(max_len, "N")
    b = seq_b.upper().ljust(max_len, "N")
    mismatches = sum(ca != cb for ca, cb in zip(a, b))
    return mismatches / max_len


def cluster_pool(
    pool: Dict[str, str],
    threshold: float = 0.20,
) -> Dict[str, List[str]]:
    """
    Greedy single-linkage clustering by Hamming distance.

    All sequences within `threshold` (normalised Hamming) of the cluster
    representative are merged into that cluster.  The representative is
    the first sequence encountered in pool insertion order.

    Scientific rationale: clustering reveals which sequence families
    dominate the pool across rounds — an enriched cluster indicates
    convergent selection toward a specific motif.

    Returns: {representative_seq_id: [member_seq_ids]}

    APPROXIMATION: greedy single-linkage is O(N²) in comparisons; for
    large pools (>1000) consider using scipy hierarchical clustering on a
    pre-computed distance matrix.  For pools ≤500 this is fast enough.
    """
    ids   = list(pool.keys())
    seqs  = list(pool.values())
    n     = len(ids)
    assigned = [-1] * n   # cluster label (index of representative)
    clusters: Dict[str, List[str]] = {}

    for i in range(n):
        if assigned[i] != -1:
            continue
        # Start new cluster with seq i as representative
        rep_id = ids[i]
        clusters[rep_id] = [rep_id]
        assigned[i] = i
        for j in range(i + 1, n):
            if assigned[j] != -1:
                continue
            if hamming_fraction(seqs[i], seqs[j]) <= threshold:
                clusters[rep_id].append(ids[j])
                assigned[j] = i

    return clusters


def cluster_enrichment_stats(
    cluster_history: List[Dict[str, List[str]]],
) -> List[Dict]:
    """
    Compute cluster size statistics across rounds.

    cluster_history[round] = {rep_id: [member_ids]} mapping.
    Returns a flat list of {round, rep_id, cluster_size, fraction} dicts
    suitable for saving as a CSV.
    """
    rows = []
    for rnd, clusters in enumerate(cluster_history):
        total = sum(len(v) for v in clusters.values())
        for rep, members in sorted(clusters.items(), key=lambda x: -len(x[1])):
            rows.append({
                "round":        rnd,
                "rep_seq_id":   rep,
                "cluster_size": len(members),
                "fraction":     round(len(members) / max(total, 1), 4),
            })
    return rows


# ─── Lineage Tracker ──────────────────────────────────────────────────────────

class LineageTracker:
    """
    Record parent → child relationships across SELEX rounds.

    Each mutation event generates a child sequence from a parent.  Tracking
    lineages allows reconstruction of evolutionary trajectories and
    identification of highly productive parent sequences.

    Usage:
        tracker = LineageTracker()
        tracker.record("R00_S00001", parent_id=None, round_num=0)
        tracker.record("R01_M00003", parent_id="R00_S00001", round_num=1)
        tracker.save("results/lineage.csv")
    """

    def __init__(self):
        self._records: List[Dict] = []
        self._seen: set = set()

    def record(
        self,
        seq_id:    str,
        round_num: int,
        parent_id: Optional[str] = None,
        sequence:  Optional[str] = None,
    ) -> None:
        if seq_id in self._seen:
            return
        self._seen.add(seq_id)
        self._records.append({
            "seq_id":    seq_id,
            "parent_id": parent_id if parent_id else "",
            "round":     round_num,
            "sequence":  sequence if sequence else "",
        })

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        fields = ["seq_id", "parent_id", "round", "sequence"]
        with open(path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fields)
            writer.writeheader()
            writer.writerows(self._records)
        log.info(f"Lineage saved: {path}  ({len(self._records)} entries)")

    def load(self, path: str) -> None:
        """Load existing lineage CSV to resume from a prior run."""
        if not os.path.exists(path):
            return
        with open(path) as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                self._records.append(row)
                self._seen.add(row["seq_id"])

    def __len__(self) -> int:
        return len(self._records)


# ─── Approximate 3D Structure Builder (A-form fallback) ──────────────────────
# A-form RNA helix parameters (Saenger, "Principles of Nucleic Acid Structure")
_RISE   = 2.81   # Å per residue along helix axis
_TWIST  = 32.70  # degrees per residue
_RADIUS = 9.0    # Å — distance from helix axis to backbone P


def build_rna_3d_approximate(sequence: str, structure: str) -> str:
    """
    Build an approximate coarse-grain 3D RNA PDB from sequence + dot-bracket.

    APPROXIMATION — OUTPUT IS NOT A REAL PREDICTED STRUCTURE.
    Used only as a fallback when RNAComposer is unavailable.
    Provides a geometrically plausible compact shape for comparative docking.
    Replace with RNAComposer / SimRNA / FARFAR2 output for publication work.
    """
    n = len(sequence)
    assert len(structure) == n, "Sequence and structure lengths must match"

    pairs   = parse_dot_bracket(structure)
    pair_map = {}
    for i, j in pairs:
        pair_map[i] = j
        pair_map[j] = i

    coords = np.zeros((n, 3))
    for i in range(n):
        angle = np.radians(i * _TWIST)
        if i in pair_map and i > pair_map[i]:
            angle += np.pi
        coords[i] = [_RADIUS * np.cos(angle), _RADIUS * np.sin(angle), i * _RISE]

    coords -= coords.mean(axis=0)
    return _coords_to_pdb(sequence, coords)


def _coords_to_pdb(sequence: str, coords: np.ndarray) -> str:
    """Format a coordinate array as PDB string with 3 atoms per nucleotide."""
    RESNAME = {"A": "  A", "U": "  U", "G": "  G", "C": "  C"}
    lines = [
        "REMARK Approximate A-form RNA (coarse-grain, digital_selex pipeline v2)",
        "REMARK 3 atoms per nucleotide: P, C4', N1/N9",
        "REMARK NOT crystallographic — for relative docking comparison only",
    ]
    atom_num = 1
    for i, (base, xyz) in enumerate(zip(sequence, coords)):
        resname = RESNAME.get(base.upper(), "  A")
        resnum  = i + 1
        x, y, z = xyz

        def fmt_atom(name, elem, dx=0.0, dy=0.0, dz=0.0):
            return (
                f"ATOM  {atom_num:5d} {name:<4s} {resname} A{resnum:4d}    "
                f"{x+dx:8.3f}{y+dy:8.3f}{z+dz:8.3f}  1.00  0.00          {elem:>2s}"
            )

        lines.append(fmt_atom(" P  ", " P"))
        atom_num += 1
        lines.append(fmt_atom(" C4'", " C", 1.5, 0.5, 1.0))
        atom_num += 1
        gly = "N9" if base.upper() in ("A", "G") else "N1"
        lines.append(fmt_atom(f" {gly} ", " N", 2.5, 1.0, 1.5))
        atom_num += 1

    lines.append("END")
    return "\n".join(lines)
