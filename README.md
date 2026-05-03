<div align="center">

# Digital SELEX v2

**In-silico aptamer discovery against TCA cycle metabolites**

[![Python 3.10](https://img.shields.io/badge/Python-3.10-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![ViennaRNA](https://img.shields.io/badge/ViennaRNA-≥2.6-2E8B57?style=flat-square)](https://www.tbi.univie.ac.at/RNA/)
[![AutoDock Vina](https://img.shields.io/badge/AutoDock_Vina-1.2.5-E76F51?style=flat-square)](https://vina.scripps.edu/)
[![PyTorch](https://img.shields.io/badge/PyTorch-≥2.0-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Nextflow](https://img.shields.io/badge/Nextflow-DSL2-0DC09D?style=flat-square)](https://nextflow.io/)
[![Status](https://img.shields.io/badge/Status-Research_Prototype-yellow?style=flat-square)]()

<br/>

*A fully automated computational pipeline that simulates SELEX — Systematic Evolution of Ligands by Exponential Enrichment — entirely in silico, targeting eight metabolites of the citric acid cycle.*

<br/>

> **Scope**: All aptamer rankings are computational predictions. Experimental validation (SPR/ITC, EMSA, NMR or cryo-EM) is required before drawing any biological conclusions.

</div>

---

## What this does

SELEX is a directed evolution method for discovering nucleic acid aptamers that bind a target molecule. Running it wet-lab costs weeks and reagents. This pipeline runs the same selection logic computationally — folding RNA sequences, docking them against metabolite targets, scoring the poses, evolving the pool, and training a surrogate model to pre-screen candidates before expensive docking steps.

Version 2 adds proper 3D structures via the RNAComposer API, negative selection against off-targets, an adaptive mutation schedule, full lineage tracking, and a CNN with self-attention for pre-screening. The pipeline is configurable and scales from a 10-sequence demo on a laptop to a 1000-sequence research run on a compute cluster.

**Targets**: citrate · isocitrate · α-ketoglutarate · succinate · fumarate · malate · oxaloacetate · succinyl-CoA

---

## Pipeline overview

```
Random RNA Pool (N=500)
        │
        ▼
  ViennaRNA — MFE 2D folding
        │  dot-bracket + MFE
        ▼
  ┌─────────────────────────────┐
  │  RNA 3D structure generator │
  │  PRIMARY : RNAComposer API  │──► PDB
  │  FALLBACK: A-form approx    │
  └─────────────────────────────┘
        │  PDB → PDBQT (meeko/RDKit)
        ▼
  AutoDock Vina (parallel)
  ├── On-targets  (7 TCA metabolites)
  └── Off-targets (negative selection)
        │
        ▼
  ┌──────────────────────────────────────────────────┐
  │  Composite scorer  — all components → [0, 1]     │
  │                                                  │
  │  binding_score     × w_aff   (Vina affinity)     │
  │  stability_score   × w_mfe   (ViennaRNA MFE)     │
  │  gc_score          × w_gc    (GC content)        │
  │  hbond_score       × w_hb    (H-bond count)      │
  │  contact_score     × w_ct    (close contacts)    │
  │  specificity_score × w_sp    (on/off-target)     │
  └──────────────────────────────────────────────────┘
        │
        ▼
  SELEX evolution (N rounds)
  ├── Select top 30%
  ├── Mutate  (adaptive μ: 0.15 → 0.05)
  ├── Inject 20% fresh random sequences
  ├── Track lineage       → results/lineage.csv
  └── Cluster by Hamming  → results/cluster_enrichment.csv
        │
        ▼
  CNN + self-attention surrogate model
  └── Pre-screens top 10% for docking only
        │
        ▼
  Ranked aptamers + 10 diagnostic plots
```

---

## What changed in v2

| Component | v1 | v2 |
|---|---|---|
| 3D structure | A-form approximation only | RNAComposer API (fragment assembly) + A-form fallback |
| Conformers per sequence | 1 | 1–5 (configurable) |
| Scoring components | 3 raw | 6 normalised, all → [0, 1] |
| Rescoring | None | H-bond count + contact count from pose geometry |
| Negative selection | None | Off-target docking → specificity score |
| Mutation rate | Fixed 0.10 | Adaptive: 0.15 (early rounds) → 0.05 (late rounds) |
| Lineage tracking | None | Full parent → child CSV |
| Sequence clustering | None | Greedy Hamming clustering, enrichment tracked per round |
| ML architecture | CNN only | CNN + self-attention + structural feature vector |
| ML role | Scoring | Pre-screening only (top 10% forwarded to docking) |
| Docking | Sequential | Parallel via `multiprocessing.Pool` |
| Pool size | 10 (demo) | 500 (configurable) |
| Diagnostic plots | 6 | 10 |

---

## Project structure

```
digital_selex/
├── config/
│   ├── config.yaml          ← all tunable parameters
│   └── targets.yaml         ← TCA metabolite SMILES + PubChem CIDs
├── structure/
│   └── rna_3d.py            ← RNAComposer API + conformer generation
├── scoring/
│   └── rescore.py           ← H-bond count + contact count from PDBQTs
├── scripts/
│   ├── 01_fetch_ligands.py
│   ├── 02_generate_sequences.py
│   ├── 03_predict_2d_structure.py
│   ├── 04_generate_3d_structure.py    ← uses structure/rna_3d.py
│   ├── 05_prepare_docking.py
│   ├── 06_run_docking.py              ← parallel (multiprocessing.Pool)
│   ├── 07_score_aptamers.py           ← 6 components + negative selection
│   ├── 08_selex_iteration.py          ← adaptive μ, lineage, clustering
│   ├── 09_train_model.py              ← CNN + self-attention
│   ├── 10_visualize_results.py        ← 10 diagnostic plots
│   └── utils/
│       ├── rna_utils.py
│       └── docking_utils.py
├── workflow/
│   ├── main.nf              ← Nextflow DSL2
│   └── nextflow.config
├── results/
│   ├── plots/
│   ├── rankings/
│   ├── full_history.csv
│   ├── lineage.csv
│   └── cluster_enrichment.csv
├── models/
├── environment.yml
└── run_pipeline.sh
```

---

## Setup

### 1. Create the conda environment

```bash
cd digital_selex
conda env create -f environment.yml
conda activate digital_selex
```

### 2. Verify key dependencies

```bash
python3 -c "import RNA; print('ViennaRNA OK')"
python3 -c "from vina import Vina; print('Vina OK')"
python3 -c "from rdkit import Chem; print('RDKit OK')"
python3 -c "import meeko; print('meeko OK')"
python3 -c "import torch; print('PyTorch OK')"
```

---

## Running the pipeline

### Quick demo (~5–15 min on a laptop)

```bash
conda activate digital_selex
./run_pipeline.sh --quick
```

`--quick` sets pool = 10, rounds = 2, exhaustiveness = 2, and uses A-form 3D structures. Use this to verify the installation works end-to-end.

### Full research run

```bash
./run_pipeline.sh
```

Default config: pool = 500, 3 rounds, exhaustiveness = 4. Expect 4–12 hours on a 4-core laptop. For publication-grade results, run on a compute cluster with exhaustiveness = 8.

### Custom flags

```bash
./run_pipeline.sh --rounds 5 --pool 200
./run_pipeline.sh --skip-ligands      # ligands already in data/ligands/
./run_pipeline.sh --no-train          # skip CNN training
```

### Step-by-step execution

```bash
# Fetch TCA metabolite 3D structures from PubChem
python3 scripts/01_fetch_ligands.py

# Run the full SELEX loop
python3 scripts/08_selex_iteration.py --rounds 3

# Train CNN + attention surrogate model
python3 scripts/09_train_model.py

# Generate diagnostic plots
python3 scripts/10_visualize_results.py
```

---

## Configuration

All parameters live in `config/config.yaml`. Key knobs:

| Section | Parameter | Default | Notes |
|---|---|---|---|
| `selex` | `initial_pool_size` | 500 | Sequences per round |
| `selex` | `early_mutation_rate` | 0.15 | Round 0 mutation rate |
| `selex` | `late_mutation_rate` | 0.05 | Final round mutation rate |
| `selex` | `selection_fraction` | 0.30 | Top fraction carried forward |
| `selex` | `cluster_threshold` | 0.20 | Hamming distance cutoff |
| `docking` | `exhaustiveness` | 4 | Vina sampling depth |
| `docking` | `n_workers` | 4 | Parallel docking processes |
| `scoring` | `use_rescore` | true | Enable geometric rescoring |
| `scoring` | `use_specificity` | true | Enable negative selection |
| `scoring` | `hbond_weight` | 0.30 | Weight on H-bond score |
| `scoring` | `specificity_weight` | 0.25 | Weight on specificity score |
| `structure3d` | `method` | `rnacomposer` | `rnacomposer` or `aform_approx` |
| `structure3d` | `n_conformers` | 3 | Conformers per sequence |
| `ml` | `use_attention` | true | Enable self-attention head |
| `ml` | `prescreening_top_fraction` | 0.10 | Fraction forwarded to docking |

For production runs: set `initial_pool_size = 1000` and `exhaustiveness = 8`.

---

## Scoring function

All six components are normalised to [0, 1] before weighting:

```
binding_score     = min-max( −Vina_affinity )
stability_score   = min-max( −MFE )
gc_score          = 1 − |GC − 0.5| / 0.5
hbond_score       = min-max( H-bond count from pose geometry )   [approximation]
contact_score     = min-max( close contacts < 4.0 Å )           [approximation]
specificity_score = min-max( mean_on_target / mean_off_target )

composite = w_aff × binding_score
          + w_mfe × stability_score
          + w_gc  × gc_score
          + w_hb  × hbond_score
          + w_ct  × contact_score
          + w_sp  × specificity_score
```

The specificity score penalises aptamers that bind off-targets nearly as well as on-targets, simulating the counter-selection step in wet-lab SELEX.

---

## ML surrogate model

The model is trained on scored aptamers from all completed SELEX rounds. Its only role is pre-screening: the top `prescreening_top_fraction` (default 10%) of new candidates pass to docking; the rest are dropped.

```
Input: one-hot RNA sequence (4 × max_len)
    │
    ├── Conv1d(4→64, k=5) → ReLU → Conv1d(64→128, k=5) → ReLU
    │       ├── GlobalMaxPool  → (128,)
    │       └── MultiheadAttention (4 heads) → GlobalAvgPool → (128,)
    │
    └── concat → (256 + n_struct_feats,)
                      │
               GC content, MFE, n_stems, n_pairs,
               n_loops, stem_fraction, loop_fraction
                      │
               Linear(hidden) → ReLU → Dropout → Linear(1)
```

> **Limitation**: training data is small (500–1500 samples after 3 rounds). The model provides useful relative rankings but is not a physical predictor of binding affinity.

---

## Outputs

```
results/
├── rankings/
│   ├── round_NN_ranked.csv             ← ranked aptamers per round
│   └── round_NN_scores_detailed.csv    ← per-aptamer × per-target breakdown
└── plots/
    ├── 01_score_distributions.png      ← violin plot per round
    ├── 02_convergence.png              ← mean score ± std across rounds
    ├── 03_heatmap_top20.png            ← top-20 aptamers × all targets
    ├── 04_gc_vs_score.png
    ├── 05_mfe_vs_affinity.png
    ├── 06_training_loss.png
    ├── 07_score_components.png         ← per-component bar chart
    ├── 08_lineage_depth.png            ← lineage depth histogram
    ├── 09_cluster_enrichment.png       ← cluster fractions across rounds
    ├── 10_convergence_cov.png          ← coefficient of variation
    ├── final_top20_aptamers.csv
    └── round_summary_stats.csv

results/full_history.csv               ← all rounds merged
results/lineage.csv                    ← parent → child relationships
results/cluster_enrichment.csv

models/
├── best_model.pt
├── model_config.json
└── training_history.csv
```

---

## Known limitations

These are not disclaimers — they describe real constraints that matter when interpreting results.

**RNA 3D structure**: RNAComposer produces fragment-assembled models, not energy-minimised structures. Loop conformations are frequently wrong. The A-form fallback is a geometric approximation with no physics. Neither method includes 2'-OH hydrogens or Mg²⁺ coordination, both of which affect RNA folding in solution.

**Docking**: AutoDock Vina was parametrised on protein-ligand complexes. RNA-specific force field terms do not exist in the current release. RNA flexibility is not modelled. Vina scores are comparative rankings, not ΔG values.

**Rescoring**: H-bond assignment uses AutoDock atom types without a geometric angle criterion. Contact counts are a rough proxy for buried surface area. Neither metric has been calibrated against experimental binding data.

**SELEX simulation**: Three computational rounds cover roughly 1500 sequences. Real SELEX explores ~10²⁵ sequences over 8–15 rounds, with PCR amplification and physical partitioning — none of which are modelled here.

**ML model**: Likely to overfit at this training set size. The self-attention layer adds marginal benefit below ~5000 training samples.

**Experimental validation required**: SPR or ITC for binding affinity, EMSA for selectivity, NMR or cryo-EM for structure. This pipeline does not substitute for any of these.

---

## Dependencies

| Tool | Version | Source |
|---|---|---|
| Python | 3.10 | conda-forge |
| ViennaRNA | ≥ 2.6 | conda-forge |
| RDKit | ≥ 2023.03 | conda-forge |
| AutoDock Vina (Python) | 1.2.5 | pip |
| meeko | 0.5.0 | pip |
| PyTorch | ≥ 2.0 | pip |
| requests | ≥ 2.31 | conda-forge |
| pandas | ≥ 2.0 | conda-forge |
| numpy | ≥ 1.24 | conda-forge |
| matplotlib | ≥ 3.7 | conda-forge |
| seaborn | ≥ 0.12 | conda-forge |
| biopython | ≥ 1.81 | conda-forge |
| Nextflow | ≥ 23.04 | nextflow.io |

---

## References

- Tuerk C, Gold L (1990). Systematic evolution of ligands by exponential enrichment. *Science* 249, 505–510.
- Antczak M et al. (2014). RNAComposer: a web server for RNA 3D structure prediction. *Nucleic Acids Res* 42(W1), W155–W159.
- Lorenz R et al. (2011). ViennaRNA Package 2.0. *Algorithms Mol Biol* 6, 26.
- Eberhardt J et al. (2021). AutoDock Vina 1.2.0. *J Chem Inf Model* 61, 3891–3898.
- Proske D et al. (2005). Aptamers — basic research, drug development, and clinical applications. *Appl Microbiol Biotechnol* 69, 367–374.

---

<div align="center">

*Pipeline v2.0.0 — computational research prototype — requires experimental validation before biological interpretation*

</div>
