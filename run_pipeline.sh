#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# run_pipeline.sh — Digital SELEX v2 end-to-end runner
#
# Usage:
#   chmod +x run_pipeline.sh
#   ./run_pipeline.sh
#
# Optional flags:
#   --rounds N          Override number of SELEX rounds
#   --pool N            Override initial pool size
#   --skip-ligands      Skip ligand fetching (use existing data/ligands/*.sdf)
#   --skip-selex        Skip SELEX; go straight to training + visualisation
#   --no-train          Skip ML model training
#   --quick             Demo mode: pool=10, rounds=2, exhaustiveness=2
#
# New in v2:
#   • RNAComposer 3D structures (config: structure3d.method=rnacomposer)
#   • Normalised multi-component scoring (binding + stability + H-bond +
#     contact + specificity)
#   • Adaptive mutation rate (μ_early=0.15 → μ_late=0.05)
#   • Lineage tracking  → results/lineage.csv
#   • Cluster enrichment → results/cluster_enrichment.csv
#   • Full history      → results/full_history.csv
#   • Parallel docking  → docking.n_workers workers
#   • CNN + self-attention ML model with structural features
#   • 10 diagnostic plots
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

# ── Activate the digital_selex conda environment ──────────────────────────────
# Works whether the script is called from (base) or any other environment.
CONDA_SH="/apps/compilers/anaconda3-2025/etc/profile.d/conda.sh"
if [ -f "${CONDA_SH}" ]; then
    # shellcheck disable=SC1090
    source "${CONDA_SH}"
    conda activate digital_selex
else
    echo "WARNING: conda init script not found at ${CONDA_SH}"
    echo "         Proceeding with current environment (may be missing packages)"
fi

ROUNDS=""
POOL=""
SKIP_LIGANDS=false
SKIP_SELEX=false
NO_TRAIN=false
QUICK=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --rounds)       ROUNDS="$2";      shift 2 ;;
        --pool)         POOL="$2";        shift 2 ;;
        --skip-ligands) SKIP_LIGANDS=true; shift ;;
        --skip-selex)   SKIP_SELEX=true;   shift ;;
        --no-train)     NO_TRAIN=true;     shift ;;
        --quick)        QUICK=true;        shift ;;
        *) echo "Unknown flag: $1"; exit 1 ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

echo "══════════════════════════════════════════════════════════"
echo "  Digital SELEX v2 — TCA Metabolite Aptamer Discovery    "
echo "══════════════════════════════════════════════════════════"
echo "  Project root: ${SCRIPT_DIR}"
echo "  Date:         $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

# ── Quick mode overrides ──────────────────────────────────────────────────────
if [ "$QUICK" = true ]; then
    echo "  [QUICK MODE] pool=10, rounds=2, exhaustiveness=2"
    python3 -c "
import yaml, sys
with open('config/config.yaml') as f: cfg = yaml.safe_load(f)
cfg['selex']['initial_pool_size'] = 10
cfg['selex']['rounds'] = 2
cfg['docking']['exhaustiveness'] = 2
cfg['docking']['n_workers'] = 1
cfg['structure3d']['method'] = 'aform_approx'
with open('config/config.yaml','w') as f: yaml.dump(cfg, f, default_flow_style=False)
print('  Quick-mode config written')
"
fi

# ── Environment check ─────────────────────────────────────────────────────────
echo "── Checking environment …"
python3 -c "import RNA"         2>/dev/null && echo "  ✓ ViennaRNA" \
    || { echo "  ✗ ViennaRNA missing. Run: conda install -c conda-forge viennarna"; exit 1; }
python3 -c "from rdkit import Chem" 2>/dev/null && echo "  ✓ RDKit" \
    || { echo "  ✗ RDKit missing.      Run: conda install -c conda-forge rdkit"; exit 1; }
python3 -c "from vina import Vina" 2>/dev/null && echo "  ✓ AutoDock Vina" \
    || { echo "  ✗ vina missing.       Run: pip install vina"; exit 1; }
python3 -c "import meeko" 2>/dev/null && echo "  ✓ meeko" \
    || { echo "  ✗ meeko missing.      Run: pip install meeko"; exit 1; }
python3 -c "import torch" 2>/dev/null && echo "  ✓ PyTorch" \
    || { echo "  ✗ PyTorch missing.    Run: pip install torch"; exit 1; }
python3 -c "import requests" 2>/dev/null && echo "  ✓ requests" \
    || { echo "  ✗ requests missing.   Run: pip install requests"; exit 1; }
echo ""

# ── Create output directories ─────────────────────────────────────────────────
mkdir -p logs data/ligands sequences structures/{2d,3d} \
         docking/{grid_boxes,pdbqt/{ligands,receptors},results} \
         models results/{plots,rankings} structure scoring

# ── Step 1: Fetch ligands ─────────────────────────────────────────────────────
if [ "$SKIP_LIGANDS" = false ]; then
    echo "── Step 1: Fetching TCA metabolite structures from PubChem …"
    python3 scripts/01_fetch_ligands.py
    echo "   Done."
else
    echo "── Step 1: Skipped (--skip-ligands)"
fi

# ── Steps 2–8: SELEX loop ─────────────────────────────────────────────────────
if [ "$SKIP_SELEX" = false ]; then
    echo ""
    echo "── Steps 2–8: Running iterative SELEX v2 loop …"
    SELEX_ARGS=""
    [ -n "$ROUNDS" ] && SELEX_ARGS="$SELEX_ARGS --rounds $ROUNDS"
    python3 scripts/08_selex_iteration.py $SELEX_ARGS
    echo "   SELEX loop complete."
else
    echo "── Steps 2–8: Skipped (--skip-selex)"
fi

# ── Step 9: Train ML model ────────────────────────────────────────────────────
if [ "$NO_TRAIN" = false ]; then
    echo ""
    echo "── Step 9: Training CNN + attention binding-score predictor …"
    python3 scripts/09_train_model.py
    echo "   Training complete."
else
    echo "── Step 9: Skipped (--no-train)"
fi

# ── Step 10: Visualise ────────────────────────────────────────────────────────
echo ""
echo "── Step 10: Generating visualisations …"
python3 scripts/10_visualize_results.py
echo "   Visualisations saved to results/plots/"

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "══════════════════════════════════════════════════════════"
echo "  Pipeline v2 Complete"
echo "══════════════════════════════════════════════════════════"
echo ""
echo "  Key outputs:"
echo "    Ranked aptamers:   results/rankings/round_*_ranked.csv"
echo "    Top-20 aptamers:   results/plots/final_top20_aptamers.csv"
echo "    Full history:      results/full_history.csv"
echo "    Lineage:           results/lineage.csv"
echo "    Cluster stats:     results/cluster_enrichment.csv"
echo "    Round summary:     results/plots/round_summary_stats.csv"
echo "    Binding heatmap:   results/plots/03_heatmap_top20.png"
echo "    Score components:  results/plots/07_score_components.png"
echo "    Convergence CoV:   results/plots/10_convergence_cov.png"
echo "    Full log:          logs/pipeline.log"
echo ""
echo "  IMPORTANT: All aptamer rankings are COMPUTATIONAL only."
echo "  Experimental validation (SPR, EMSA, NMR) is required"
echo "  before any biological conclusion can be drawn."
echo ""
echo "  Finished: $(date '+%Y-%m-%d %H:%M:%S')"
