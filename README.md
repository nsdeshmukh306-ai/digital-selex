# Digital SELEX v2 — In-silico Aptamer Discovery (Research-Grade Prototype)

A fully automated computational pipeline that simulates the SELEX
(Systematic Evolution of Ligands by Exponential Enrichment) process
entirely in-silico, targeting eight Citric Acid Cycle metabolites.

**This is a research-grade prototype for computational and educational use.
All aptamer rankings are computational only. Experimental validation is
required before any biological conclusion can be drawn.**

---

## Pipeline Diagram (v2)

```
Random RNA Pool (N=500)
         │
         ▼
  ViennaRNA — MFE 2D Folding
         │  dot-bracket + MFE
         ▼
  ┌──────────────────────────────┐
  │  RNA 3D Structure Generator  │
  │  PRIMARY: RNAComposer API    │ ─── PDB
  │  FALLBACK: A-form approx     │
  └──────────────────────────────┘
         │  PDB → PDBQT (meeko/RDKit)
         ▼
  AutoDock Vina (parallel, N workers)
  │  On-targets (7 metabolites)
  │  Off-targets (negative selection)
         │
         ▼
  ┌────────────────────────────────────────────────────┐
  │  Composite Scorer (normalised, all → [0,1])        │
  │  binding_score    × w_aff  (Vina affinity)         │
  │  stability_score  × w_mfe  (MFE, ViennaRNA)        │
  │  gc_score         × w_gc   (GC content)            │
  │  hbond_score      × w_hb   (H-bond count, rescore) │
  │  contact_score    × w_ct   (close contacts 4Å)     │
  │  specificity_score× w_sp   (on/off-target ratio)   │
  └────────────────────────────────────────────────────┘
         │
         ▼
  SELEX Evolution (N rounds)
  ├── Select top 30%
  ├── Mutate (adaptive μ: 0.15 → 0.05)
  ├── Inject 20% fresh random sequences
  ├── Track lineage → results/lineage.csv
  └── Cluster by Hamming distance → results/cluster_enrichment.csv
         │
         ▼
  CNN + Self-Attention Surrogate Model
  │  Input: one-hot sequence + structural features (GC, MFE, stems, loops)
  │  Use: pre-screen top 10% for docking only
         │
         ▼
  Ranked Aptamers + 10 Diagnostic Plots
```

**Targets**: citrate · isocitrate · α-ketoglutarate · succinate ·
fumarate · malate · oxaloacetate · succinyl-CoA

---

## What's New in v2

| Component | v1 | v2 |
|---|---|---|
| 3D structure | A-form approx only | RNAComposer API (fragment-assembly) + A-form fallback |
| Conformers | 1 | 1–5 (configurable) |
| Scoring | 3 raw components | 6 normalised components, all → [0,1] |
| Rescoring | None | H-bond count + contact count from pose geometry |
| Negative selection | None | Off-target docking → specificity score |
| Mutation rate | Fixed 0.10 | Adaptive: 0.15 (early) → 0.05 (late) |
| Lineage tracking | None | Full parent→child CSV |
| Clustering | None | Greedy Hamming clustering, enrichment tracked |
| ML model | CNN only | CNN + self-attention + structural feature vector |
| ML use | Scoring | Pre-screening only (top 10% → docking) |
| Docking | Sequential | Parallel (configurable n_workers) |
| Pool size | 10 (demo) | 500 (configurable) |
| Plots | 6 | 10 |
| Outputs | ranked CSVs | + lineage + cluster enrichment + full_history + round_summary |

---

## Project Structure

```
digital_selex/
├── config/
│   ├── config.yaml          ← all tunable parameters (expanded in v2)
│   └── targets.yaml         ← TCA metabolite SMILES + PubChem CIDs
├── structure/               ← NEW: RNA 3D structure module
│   └── rna_3d.py            ← RNAComposer API + conformer generation
├── scoring/                 ← NEW: geometric rescoring module
│   └── rescore.py           ← H-bond count + contact count from PDBQTs
├── data/
│   └── ligands/             ← metabolite SDF + SMILES
├── sequences/               ← FASTA + CSV pools per round
├── structures/
│   ├── 2d/                  ← dot-bracket + MFE per sequence
│   └── 3d/                  ← PDB files per sequence (+ conformers)
├── docking/
│   ├── grid_boxes/          ← Vina grid JSON configs
│   ├── pdbqt/               ← PDBQT files for ligands and receptors
│   └── results/             ← Vina output poses + score CSVs
├── models/                  ← PyTorch weights + training history
├── results/
│   ├── plots/               ← 10 PNG visualisations
│   ├── rankings/            ← scored + ranked aptamer CSVs
│   ├── full_history.csv     ← all rounds merged (NEW)
│   ├── lineage.csv          ← parent→child relationships (NEW)
│   └── cluster_enrichment.csv ← cluster sizes per round (NEW)
├── scripts/
│   ├── utils/
│   │   ├── rna_utils.py     ← expanded: features, lineage, clustering
│   │   └── docking_utils.py ← unchanged
│   ├── 01_fetch_ligands.py
│   ├── 02_generate_sequences.py
│   ├── 03_predict_2d_structure.py
│   ├── 04_generate_3d_structure.py  ← uses structure/rna_3d.py
│   ├── 05_prepare_docking.py
│   ├── 06_run_docking.py            ← parallel (multiprocessing.Pool)
│   ├── 07_score_aptamers.py         ← 6 normalised components + neg. sel.
│   ├── 08_selex_iteration.py        ← adaptive μ, lineage, clustering
│   ├── 09_train_model.py            ← CNN + self-attention + struct features
│   └── 10_visualize_results.py      ← 10 diagnostic plots
├── workflow/
│   ├── main.nf              ← Nextflow DSL2 (round 0 only)
│   └── nextflow.config
├── environment.yml
├── run_pipeline.sh          ← updated v2 runner
└── README.md
```

---

## Setup

### 1. Create conda environment

```bash
cd digital_selex
conda env create -f environment.yml
conda activate digital_selex
```

### 2. Verify installation

```bash
python3 -c "import RNA; print('ViennaRNA OK')"
python3 -c "from vina import Vina; print('Vina OK')"
python3 -c "from rdkit import Chem; print('RDKit OK')"
python3 -c "import meeko; print('meeko OK')"
python3 -c "import torch; print('PyTorch OK')"
python3 -c "import requests; print('requests OK')"
```

---

## Running the Pipeline

### Quick demo (≈5–15 min on laptop)

```bash
conda activate digital_selex
cd digital_selex
./run_pipeline.sh --quick
```

`--quick` sets pool=10, rounds=2, exhaustiveness=2, A-form 3D structures.

### Full research-grade run

```bash
conda activate digital_selex
cd digital_selex
./run_pipeline.sh
```

With default config (pool=500, 3 rounds, exhaustiveness=4):
expect 4–12 hours on a 4-core laptop.
Use a compute cluster with exhaustiveness=8 for publication-grade results.

### Custom run

```bash
./run_pipeline.sh --rounds 5 --pool 200
./run_pipeline.sh --skip-ligands      # ligands already in data/ligands/
./run_pipeline.sh --no-train          # skip CNN training
```

### Step-by-step

```bash
# 1. Fetch metabolite 3D structures
python3 scripts/01_fetch_ligands.py

# 2-8. Full SELEX loop (recommended entry point)
python3 scripts/08_selex_iteration.py --rounds 3

# 9. Train CNN + attention model
python3 scripts/09_train_model.py

# 10. Plots + summaries
python3 scripts/10_visualize_results.py
```

---

## Configuration (`config/config.yaml`)

### Key parameters

| Section | Parameter | Default | Description |
|---|---|---|---|
| selex | initial_pool_size | 500 | Sequences per round |
| selex | early_mutation_rate | 0.15 | Mutation rate in round 0 |
| selex | late_mutation_rate | 0.05 | Mutation rate in final round |
| selex | selection_fraction | 0.30 | Top fraction selected |
| selex | cluster_threshold | 0.20 | Hamming distance for clustering |
| docking | exhaustiveness | 4 | Vina sampling depth |
| docking | n_workers | 4 | Parallel docking processes |
| scoring | use_rescore | true | Enable H-bond/contact rescoring |
| scoring | use_specificity | true | Enable negative selection |
| scoring | off_targets | [] | Off-target metabolite names |
| scoring | hbond_weight | 0.30 | Weight on H-bond score |
| scoring | specificity_weight | 0.25 | Weight on specificity score |
| structure3d | method | rnacomposer | '`rnacomposer`' or '`aform_approx`' |
| structure3d | n_conformers | 3 | Conformers per sequence |
| ml | use_attention | true | CNN + self-attention |
| ml | prescreening_top_fraction | 0.10 | Top ML % forwarded to docking |

For production runs: increase `initial_pool_size` to 1000 and `exhaustiveness` to 8.

---

## Scoring Function

```
All components normalised to [0, 1] before weighting.

binding_score     = min-max(-Vina_affinity)
stability_score   = min-max(-MFE)
gc_score          = 1 - |GC - 0.5| / 0.5
hbond_score       = min-max(H-bond count from pose geometry) [APPROXIMATION]
contact_score     = min-max(close contacts < 4.0 Å)         [APPROXIMATION]
specificity_score = min-max(mean_on_target / mean_off_target)

composite = w_aff × binding_score
          + w_mfe × stability_score
          + w_gc  × gc_score
          + w_hb  × hbond_score
          + w_ct  × contact_score
          + w_sp  × specificity_score

mean_score       = mean over all on-targets
specificity_index = max_score / (mean_score + ε)
```

---

## Negative Selection (Section 3)

Off-target metabolites are listed in `config.yaml → scoring.off_targets`.
They must be present in `data/ligands/` (add their SMILES to `targets.yaml`
with `skip_docking: false`; they will be docked but excluded from the
positive score calculation).

The specificity score penalises aptamers that bind off-targets nearly as
well as on-targets, simulating the counter-selection step in real SELEX.

---

## RNAComposer Integration (Section 1)

When `structure3d.method = rnacomposer`:
1. The pipeline submits each sequence + dot-bracket to the RNAComposer REST API
   (Antczak et al., NAR 2014; http://rnacomposer.cs.put.poznan.pl/).
2. It polls for the PDB result (up to 120 s, configurable).
3. If unreachable, it falls back to the A-form approximation automatically.

**APPROXIMATION**: When `n_conformers > 1`, additional conformers are
generated by adding Gaussian coordinate noise (σ = 0.5 Å) to the primary
model. These are not physics-derived conformers; they sample slightly
different docking grid presentations.

---

## ML Model (Section 6)

Architecture: 1D CNN → self-attention → MLP with structural feature fusion.

```
Input sequence (one-hot, 4 × max_len)
    │
    ├── Conv1d(4,64,k=5) → ReLU → Conv1d(64,128,k=5) → ReLU
    │       │
    │       ├── GlobalMaxPool → (128,)
    │       └── MultiheadAttention (4 heads) → GlobalAvgPool → (128,)
    │
    └── concat → (256 + n_struct_feats,)
                      │
               GC, MFE, n_stems, n_pairs,
               n_loops, stem_fraction, loop_fraction
                      │
               Linear(hidden) → ReLU → Dropout → Linear(1)
```

**Use**: the model is trained on scored aptamers from all SELEX rounds and
used ONLY to pre-screen new candidate sequences in subsequent rounds.
The top `prescreening_top_fraction` (default 10%) pass to expensive docking;
the rest are discarded.

**Limitation**: training data is small (500–1500 samples after 3 rounds).
The model provides a useful relative ranking but should not be interpreted
as a physical predictor of binding affinity.

---

## Outputs

After successful completion:

```
results/rankings/
  round_NN_ranked.csv          ← ranked aptamers per round
  round_NN_scores_detailed.csv ← per-aptamer × per-target scores

results/plots/
  01_score_distributions.png   ← violin plot per round
  02_convergence.png           ← mean score ± std across rounds
  03_heatmap_top20.png         ← top-20 × all targets
  04_gc_vs_score.png
  05_mfe_vs_affinity.png
  06_training_loss.png
  07_score_components.png      ← NEW: binding/stability/hbond/contact/specificity bars
  08_lineage_depth.png         ← NEW: lineage depth histogram
  09_cluster_enrichment.png    ← NEW: cluster fraction across rounds
  10_convergence_cov.png       ← NEW: coefficient of variation (diversity measure)
  final_top20_aptamers.csv     ← final ranked aptamer list
  round_summary_stats.csv      ← NEW: mean/std/min/max per round

results/full_history.csv       ← NEW: all rounds merged
results/lineage.csv            ← NEW: parent→child relationships
results/cluster_enrichment.csv ← NEW: cluster sizes per round

models/
  best_model.pt                ← best CNN+attention weights
  model_config.json            ← architecture + normalisation params
  training_history.csv

logs/
  pipeline.log
```

---

## LIMITATIONS AND NON-VALIDATED ASSUMPTIONS

### 1. RNA 3D Structure (CRITICAL)
- **RNAComposer** generates realistic fragment-assembled structures but
  still contains global errors for long or unusual sequences.
- **A-form fallback** is a geometric approximation with no energy minimisation.
- Loop conformations in both methods may be inaccurate.
- Structures lack explicit 2'-OH hydrogens and Mg²⁺ coordination.

### 2. Docking with AutoDock Vina
- Vina was parametrised on protein–ligand complexes; RNA-specific force
  field terms are not available.
- RNA flexibility is NOT modelled (rigid receptor).
- Divalent cation effects are ignored.
- Scores are comparative rankings, not absolute binding energies (ΔG).

### 3. Rescoring (H-bonds, contacts)
- **APPROXIMATION**: H-bond assignment uses AutoDock atom types only;
  no geometric angle criterion is applied (H positions unavailable).
- Contact counts are a rough proxy for buried surface area.
- Neither component is calibrated to experimental binding data.

### 4. Negative Selection
- Off-target metabolites must be manually added and docked.
- Specificity scores reflect in-silico docking, not true selectivity.

### 5. SELEX Simulation
- 3 computational rounds ≠ 8–15 wet-lab rounds.
- Sequence space explored: ~1,500 vs. 10²⁵ in real SELEX.
- No PCR amplification bias or partitioning errors are modelled.

### 6. ML Model
- Training set: 500–1,500 samples. Likely to overfit.
- Attention mechanism provides marginal benefit at this data scale.
- Used for pre-screening only; not an independent binding predictor.

### 7. Conformer Perturbations
- Gaussian noise perturbations are NOT physics-derived conformers.
- Used only to diversify docking pose sampling.

### 8. No Experimental Validation
- Validation requires: SPR or ITC (Kd), EMSA (binding), NMR or cryo-EM
  (structure). None of these are substituted by this pipeline.

---

## Dependencies

| Tool | Version | Source |
|------|---------|--------|
| Python | 3.10 | conda-forge |
| ViennaRNA | ≥2.6 | conda-forge |
| RDKit | ≥2023.03 | conda-forge |
| AutoDock Vina (Python) | 1.2.5 | pip |
| meeko | 0.5.0 | pip |
| PyTorch | ≥2.0 | pip |
| requests | ≥2.31 | conda-forge |
| pandas | ≥2.0 | conda-forge |
| numpy | ≥1.24 | conda-forge |
| matplotlib | ≥3.7 | conda-forge |
| seaborn | ≥0.12 | conda-forge |
| biopython | ≥1.81 | conda-forge |
| Nextflow | ≥23.04 | nextflow.io |

---

## References

- Antczak M et al. (2014) RNAComposer. NAR 42(W1):W155–W159.
- Lorenz R et al. (2011) ViennaRNA Package 2.0. Algorithms Mol. Biol. 6:26.
- Eberhardt J et al. (2021) AutoDock Vina 1.2.0. J. Chem. Inf. Model. 61:3891.
- Tuerk C, Gold L (1990) SELEX. Science 249:505–510.
- Proske D et al. (2005) Aptamers. Appl. Microbiol. Biotechnol. 69:367–374.

---

*Pipeline v2.0.0 — computational research prototype — requires experimental validation.*
