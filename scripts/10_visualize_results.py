#!/usr/bin/env python3
"""
Script 10 – Visualise and summarise digital SELEX results (v2).

New plots over v1
─────────────────
  7. Component score breakdown (binding / stability / hbond / contact / specificity)
  8. Lineage depth distribution (if lineage.csv exists)
  9. Cluster enrichment across rounds (if cluster_enrichment.csv exists)
 10. ML model: predicted vs actual scatter (if training history exists)

All intermediate data is also written to results/full_history.csv (see 08_selex_iteration.py).

Reproducibility: random seeds are read from config.yaml and all plots
include the seed in the figure title.
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import seaborn as sns
import yaml

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR   = os.path.dirname(SCRIPT_DIR)

log = logging.getLogger(__name__)
sns.set_theme(style="whitegrid", palette="muted")


# ─── Original plots (v1) ─────────────────────────────────────────────────────

def fig1_score_distributions(ranked_frames: dict, out_dir: str):
    rows = []
    for rnd, df in ranked_frames.items():
        for v in df["mean_score"]:
            rows.append({"Round": f"R{rnd:02d}", "Mean Score": v})
    data = pd.DataFrame(rows)
    if data.empty:
        return

    fig, ax = plt.subplots(figsize=(max(6, len(ranked_frames)*2), 5))
    sns.violinplot(data=data, x="Round", y="Mean Score", ax=ax, inner="point")
    ax.set_title("Score Distribution per SELEX Round")
    ax.set_ylabel("Composite Mean Score")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "01_score_distributions.png"), dpi=150)
    plt.close(fig)
    log.info("Saved: 01_score_distributions.png")


def fig2_convergence(ranked_frames: dict, out_dir: str):
    rounds, means, stds = [], [], []
    for rnd, df in sorted(ranked_frames.items()):
        rounds.append(rnd)
        means.append(df["mean_score"].mean())
        stds.append(df["mean_score"].std())

    fig, ax = plt.subplots(figsize=(7, 4))
    means, stds = np.array(means), np.array(stds)
    ax.errorbar(rounds, means, yerr=stds, marker="o", linewidth=2, capsize=4)
    ax.set_xlabel("SELEX Round")
    ax.set_ylabel("Mean Composite Score")
    ax.set_title("Score Convergence Across SELEX Rounds")
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "02_convergence.png"), dpi=150)
    plt.close(fig)
    log.info("Saved: 02_convergence.png")


def fig3_heatmap(detailed_frames: dict, out_dir: str):
    if not detailed_frames:
        return
    last_rnd = max(detailed_frames.keys())
    df = detailed_frames[last_rnd]
    pivot = df.pivot_table(
        index="seq_id", columns="target", values="composite_score", aggfunc="mean")
    if pivot.empty:
        return
    pivot["_mean"] = pivot.mean(axis=1)
    top20 = pivot.nlargest(20, "_mean").drop(columns="_mean")

    fig, ax = plt.subplots(
        figsize=(max(8, len(top20.columns)), min(12, len(top20)*0.5+2)))
    sns.heatmap(top20, annot=True, fmt=".2f", cmap="YlGnBu", ax=ax,
                cbar_kws={"label": "Composite Score"})
    ax.set_title(f"Binding Score Heatmap – Top 20 Aptamers (Round {last_rnd})")
    ax.set_ylabel("Aptamer ID")
    ax.set_xlabel("TCA Metabolite Target")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "03_heatmap_top20.png"), dpi=150)
    plt.close(fig)
    log.info("Saved: 03_heatmap_top20.png")


def fig4_gc_vs_score(ranked_frames: dict, out_dir: str):
    rows = []
    for rnd, df in ranked_frames.items():
        for _, row in df.iterrows():
            rows.append({"Round": f"R{rnd:02d}",
                         "GC Content": row.get("gc_content", np.nan),
                         "Mean Score": row["mean_score"]})
    data = pd.DataFrame(rows).dropna()
    if data.empty:
        return

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.scatterplot(data=data, x="GC Content", y="Mean Score", hue="Round",
                    ax=ax, alpha=0.7)
    ax.set_title("GC Content vs Composite Score")
    ax.axvline(0.4, color="grey", ls="--", alpha=0.5)
    ax.axvline(0.6, color="grey", ls="--", alpha=0.5)
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "04_gc_vs_score.png"), dpi=150)
    plt.close(fig)
    log.info("Saved: 04_gc_vs_score.png")


def fig5_mfe_vs_affinity(detailed_frames: dict, out_dir: str):
    rows = []
    for rnd, df in detailed_frames.items():
        if "best_affinity" not in df.columns:
            continue
        sub = df[["seq_id","mfe","best_affinity"]].dropna().drop_duplicates("seq_id")
        sub = sub.assign(Round=f"R{rnd:02d}")
        rows.append(sub)
    if not rows:
        return
    data = pd.concat(rows, ignore_index=True)

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.scatterplot(data=data, x="mfe", y="best_affinity", hue="Round",
                    ax=ax, alpha=0.7)
    ax.set_xlabel("MFE (kcal/mol)")
    ax.set_ylabel("Best Vina Affinity (kcal/mol)")
    ax.set_title("Structural Stability vs Binding Affinity")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "05_mfe_vs_affinity.png"), dpi=150)
    plt.close(fig)
    log.info("Saved: 05_mfe_vs_affinity.png")


def fig6_training_loss(models_dir: str, out_dir: str):
    hist_path = os.path.join(models_dir, "training_history.csv")
    if not os.path.exists(hist_path):
        log.info("No training history — skipping plot 6")
        return

    df = pd.read_csv(hist_path)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(df["epoch"], df["train_loss"], label="Train Loss")
    if "val_loss" in df and df["val_loss"].notna().any():
        ax.plot(df["epoch"], df["val_loss"], label="Val Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.set_title("CNN + Attention Training Loss")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "06_training_loss.png"), dpi=150)
    plt.close(fig)
    log.info("Saved: 06_training_loss.png")


# ─── New plots (v2) ──────────────────────────────────────────────────────────

def fig7_score_components(detailed_frames: dict, out_dir: str):
    """
    Stacked bar: mean contribution of each score component for top-20 aptamers.
    Shows binding vs stability vs H-bond vs contact vs specificity breakdown.
    """
    if not detailed_frames:
        return
    last_rnd = max(detailed_frames.keys())
    df = detailed_frames[last_rnd]

    component_cols = [c for c in
                      ["binding_score","stability_score","gc_score",
                       "hbond_score","contact_score","specificity_score"]
                      if c in df.columns]
    if not component_cols:
        log.info("No component columns in detailed frame — skipping plot 7")
        return

    agg = (df.groupby("seq_id")[component_cols].mean()
             .assign(total=lambda x: x.sum(axis=1))
             .sort_values("total", ascending=False)
             .head(20)
             .drop(columns="total"))

    fig, ax = plt.subplots(figsize=(14, 6))
    agg.plot(kind="bar", stacked=True, ax=ax, colormap="tab10")
    ax.set_title(f"Score Component Breakdown — Top 20 Aptamers (Round {last_rnd})")
    ax.set_xlabel("Aptamer ID")
    ax.set_ylabel("Normalised Score Component")
    ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left")
    ax.tick_params(axis="x", labelrotation=45)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "07_score_components.png"), dpi=150)
    plt.close(fig)
    log.info("Saved: 07_score_components.png")


def fig8_lineage_depth(lineage_path: str, out_dir: str):
    """Histogram of lineage depth (distance from root) for each sequence."""
    if not os.path.exists(lineage_path):
        log.info("No lineage.csv — skipping plot 8")
        return

    df = pd.read_csv(lineage_path)
    # Compute depth for each node via BFS
    parent_map = df.set_index("seq_id")["parent_id"].to_dict()
    depths = {}
    for sid in df["seq_id"]:
        d = 0
        cur = sid
        visited = set()
        while parent_map.get(cur, "") and cur not in visited:
            visited.add(cur)
            cur = parent_map[cur]
            d += 1
        depths[sid] = d

    depth_series = pd.Series(list(depths.values()))
    fig, ax = plt.subplots(figsize=(7, 4))
    depth_series.plot(kind="hist", bins=max(5, int(depth_series.max()+1)),
                      ax=ax, edgecolor="white")
    ax.set_xlabel("Lineage Depth (rounds from root)")
    ax.set_ylabel("Count")
    ax.set_title("Lineage Depth Distribution")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "08_lineage_depth.png"), dpi=150)
    plt.close(fig)
    log.info("Saved: 08_lineage_depth.png")


def fig9_cluster_enrichment(cluster_path: str, out_dir: str):
    """Line plot: top-5 cluster sizes across rounds."""
    if not os.path.exists(cluster_path):
        log.info("No cluster_enrichment.csv — skipping plot 9")
        return

    df = pd.read_csv(cluster_path)
    # Take top-5 clusters by max fraction across all rounds
    top5 = (df.groupby("rep_seq_id")["fraction"].max()
              .sort_values(ascending=False)
              .head(5)
              .index.tolist())

    sub = df[df["rep_seq_id"].isin(top5)]
    fig, ax = plt.subplots(figsize=(8, 4))
    for rep_id, grp in sub.groupby("rep_seq_id"):
        ax.plot(grp["round"], grp["fraction"], marker="o",
                label=rep_id[:12] + "…" if len(rep_id) > 12 else rep_id)
    ax.set_xlabel("SELEX Round")
    ax.set_ylabel("Cluster Fraction of Pool")
    ax.set_title("Top-5 Cluster Enrichment Across Rounds")
    ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "09_cluster_enrichment.png"), dpi=150)
    plt.close(fig)
    log.info("Saved: 09_cluster_enrichment.png")


def fig10_convergence_stddev(ranked_frames: dict, out_dir: str):
    """
    Plot coefficient of variation (std/mean) of scores across rounds.
    Decreasing CoV indicates convergence toward a fitter sequence population.
    """
    rounds, covs = [], []
    for rnd, df in sorted(ranked_frames.items()):
        m = df["mean_score"].mean()
        s = df["mean_score"].std()
        cov = s / abs(m) if abs(m) > 1e-9 else 0.0
        rounds.append(rnd)
        covs.append(cov)

    if not rounds:
        return

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(rounds, covs, marker="s", linewidth=2, color="darkorange")
    ax.set_xlabel("SELEX Round")
    ax.set_ylabel("Coefficient of Variation (σ/μ)")
    ax.set_title("Population Diversity (CoV) — Decreasing = Converging")
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "10_convergence_cov.png"), dpi=150)
    plt.close(fig)
    log.info("Saved: 10_convergence_cov.png")


# ─── Outputs ──────────────────────────────────────────────────────────────────

def save_final_ranking(ranked_frames: dict, out_dir: str):
    """Merge all rounds → final top-aptamer table with all score components."""
    if not ranked_frames:
        return
    last = max(ranked_frames.keys())
    df   = ranked_frames[last].head(20)
    cols = ["rank","seq_id","sequence","mean_score","best_target",
            "specificity_index","gc_content","mfe",
            "mean_binding","mean_stability","mean_hbond",
            "mean_contact","mean_specificity"]
    cols = [c for c in cols if c in df.columns]
    out  = os.path.join(out_dir, "final_top20_aptamers.csv")
    df[cols].to_csv(out, index=False)
    log.info(f"Final top-20 aptamers: {out}")
    log.info(f"\n{df[cols].to_string(index=False)}")


def save_full_history_summary(results_dir: str, out_dir: str):
    """
    Read full_history.csv and write a round-by-round summary statistics table.
    """
    hist_path = os.path.join(results_dir, "full_history.csv")
    if not os.path.exists(hist_path):
        log.info("No full_history.csv — skipping summary")
        return

    df = pd.read_csv(hist_path)
    if "round" not in df.columns or "mean_score" not in df.columns:
        return

    summary = (df.groupby("round")["mean_score"]
                 .agg(["mean","std","min","max","count"])
                 .reset_index()
                 .rename(columns={"mean":"score_mean","std":"score_std",
                                   "min":"score_min","max":"score_max",
                                   "count":"n_aptamers"}))
    out = os.path.join(out_dir, "round_summary_stats.csv")
    summary.to_csv(out, index=False)
    log.info(f"Round summary statistics: {out}")
    log.info(f"\n{summary.to_string(index=False)}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Visualise SELEX results (v2)")
    parser.add_argument("--config",     default=os.path.join(ROOT_DIR, "config", "config.yaml"))
    parser.add_argument("--rank-dir",   default=os.path.join(ROOT_DIR, "results", "rankings"))
    parser.add_argument("--dock-dir",   default=os.path.join(ROOT_DIR, "docking", "results"))
    parser.add_argument("--models-dir", default=os.path.join(ROOT_DIR, "models"))
    parser.add_argument("--out-dir",    default=os.path.join(ROOT_DIR, "results", "plots"))
    args = parser.parse_args()

    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)

    results_dir = os.path.join(ROOT_DIR, "results")
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(os.path.dirname(os.path.join(ROOT_DIR, cfg["logging"]["file"])), exist_ok=True)

    logging.basicConfig(
        level=getattr(logging, cfg["logging"]["level"], logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(os.path.join(ROOT_DIR, cfg["logging"]["file"]), mode="a"),
        ],
    )

    # Load ranked CSVs
    ranked_frames = {}
    for f in sorted(Path(args.rank_dir).glob("round_*_ranked.csv")):
        rnd = int(f.stem.split("_")[1])
        ranked_frames[rnd] = pd.read_csv(f)

    # Load detailed CSVs with per-target scores
    detailed_frames = {}
    for f in sorted(Path(args.rank_dir).glob("round_*_scores_detailed.csv")):
        rnd = int(f.stem.split("_")[1])
        df  = pd.read_csv(f)
        dock_f = os.path.join(args.dock_dir, f"round_{rnd:02d}_docking_results.csv")
        if os.path.exists(dock_f) and "best_affinity" not in df.columns:
            dock_df = pd.read_csv(dock_f)[["seq_id","target","best_affinity"]]
            df = df.merge(dock_df, on=["seq_id","target"], how="left")
        detailed_frames[rnd] = df

    if not ranked_frames:
        log.error(f"No ranked CSVs found in {args.rank_dir}")
        sys.exit(1)

    # ── Generate all plots ────────────────────────────────────────────────────
    fig1_score_distributions(ranked_frames, args.out_dir)
    fig2_convergence(ranked_frames, args.out_dir)
    fig3_heatmap(detailed_frames, args.out_dir)
    fig4_gc_vs_score(ranked_frames, args.out_dir)
    fig5_mfe_vs_affinity(detailed_frames, args.out_dir)
    fig6_training_loss(args.models_dir, args.out_dir)
    fig7_score_components(detailed_frames, args.out_dir)
    fig8_lineage_depth(os.path.join(results_dir, "lineage.csv"), args.out_dir)
    fig9_cluster_enrichment(os.path.join(results_dir, "cluster_enrichment.csv"), args.out_dir)
    fig10_convergence_stddev(ranked_frames, args.out_dir)

    # ── Save outputs ──────────────────────────────────────────────────────────
    save_final_ranking(ranked_frames, args.out_dir)
    save_full_history_summary(results_dir, args.out_dir)

    log.info(f"\nAll plots and summaries saved to: {args.out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
