#!/usr/bin/env nextflow
/*
 * digital_selex — Nextflow DSL2 pipeline
 *
 * Full in-silico SELEX targeting TCA cycle metabolites.
 * Run from the project root:
 *   nextflow run workflow/main.nf -c workflow/nextflow.config
 *
 * Requires:
 *   - Conda environment 'digital_selex' activated, OR
 *   - All dependencies installed in the active environment
 */

nextflow.enable.dsl = 2

// ─── Helper: absolute path to scripts directory ────────────────────────────
def scripts = params.scripts_dir


// ════════════════════════════════════════════════════════════════════════════
//  PROCESSES
// ════════════════════════════════════════════════════════════════════════════

process FETCH_LIGANDS {
    tag "fetch_ligands"
    publishDir "${params.outdir}/../data/ligands", mode: "copy", overwrite: true

    output:
    path "*.sdf", emit: sdf_files

    script:
    """
    python ${scripts}/01_fetch_ligands.py \
        --config  ${params.config_file}  \
        --targets ${params.targets_file} \
        --outdir  .
    """
}


process GENERATE_SEQUENCES {
    tag "generate_round_${round_num}"
    publishDir "${params.outdir}/../sequences", mode: "copy", overwrite: true

    input:
    val round_num

    output:
    path "round_${String.format('%02d', round_num)}_pool.fasta", emit: fasta
    path "round_${String.format('%02d', round_num)}_pool.csv",   emit: csv

    script:
    def rnd_pad = String.format('%02d', round_num)
    """
    python ${scripts}/02_generate_sequences.py \
        --config    ${params.config_file} \
        --round     ${round_num}          \
        --pool-size ${params.initial_pool_size} \
        --outdir    .
    """
}


process PREDICT_2D_STRUCTURE {
    tag "2d_struct_round_${round_num}"
    publishDir "${params.outdir}/../structures/2d", mode: "copy", overwrite: true

    input:
    path fasta_file
    val  round_num

    output:
    path "round_${String.format('%02d', round_num)}_structures.csv", emit: csv
    path "*.dbn", emit: dbn_files

    script:
    def rnd_pad = String.format('%02d', round_num)
    """
    python ${scripts}/03_predict_2d_structure.py \
        --config ${params.config_file} \
        --fasta  ${fasta_file}         \
        --outdir .                     \
        --round  ${round_num}
    """
}


process GENERATE_3D_STRUCTURE {
    tag "3d_struct_round_${round_num}"
    publishDir "${params.outdir}/../structures/3d", mode: "copy", overwrite: true

    input:
    path struct_csv
    val  round_num

    output:
    path "*.pdb", emit: pdb_files

    script:
    """
    python ${scripts}/04_generate_3d_structure.py \
        --config     ${params.config_file} \
        --struct-csv ${struct_csv}         \
        --outdir     .
    """
}


process PREPARE_DOCKING {
    tag "prep_docking_round_${round_num}"
    publishDir "${params.outdir}/../docking", mode: "copy", overwrite: true

    input:
    path pdb_files
    path sdf_files
    val  round_num

    output:
    path "pdbqt/ligands/*.pdbqt",   emit: ligand_pdbqts
    path "pdbqt/receptors/*.pdbqt", emit: receptor_pdbqts
    path "grid_boxes/*.json",        emit: grid_files

    script:
    """
    mkdir -p pdbqt/ligands pdbqt/receptors grid_boxes

    # Copy SDFs to data/ligands for the script to find
    mkdir -p ligands_tmp
    cp ${sdf_files} ligands_tmp/ 2>/dev/null || true

    python ${scripts}/05_prepare_docking.py \
        --config   ${params.config_file}  \
        --targets  ${params.targets_file} \
        --3d-dir   .                      \
        --round    ${round_num}
    """
}


process RUN_DOCKING {
    tag "docking_round_${round_num}"
    cpus   params.docking_cpu
    publishDir "${params.outdir}/../docking/results", mode: "copy", overwrite: true

    input:
    path ligand_pdbqts
    path receptor_pdbqts
    path grid_files
    val  round_num

    output:
    path "round_${String.format('%02d', round_num)}_docking_results.csv", emit: csv
    path "*_out.pdbqt", optional: true, emit: docked_poses

    script:
    """
    python ${scripts}/06_run_docking.py \
        --config  ${params.config_file}  \
        --targets ${params.targets_file} \
        --round   ${round_num}
    """
}


process SCORE_APTAMERS {
    tag "score_round_${round_num}"
    publishDir "${params.outdir}/rankings", mode: "copy", overwrite: true

    input:
    path docking_csv
    path struct_csv
    path seq_csv
    val  round_num

    output:
    path "round_${String.format('%02d', round_num)}_ranked.csv",          emit: ranked
    path "round_${String.format('%02d', round_num)}_scores_detailed.csv", emit: detailed

    script:
    """
    python ${scripts}/07_score_aptamers.py \
        --config      ${params.config_file} \
        --docking-csv ${docking_csv}        \
        --struct-csv  ${struct_csv}         \
        --seq-csv     ${seq_csv}            \
        --outdir      .                     \
        --round       ${round_num}
    """
}


process TRAIN_MODEL {
    tag "train_model"
    publishDir "${params.outdir}/../models", mode: "copy", overwrite: true

    input:
    path ranked_csvs  // all round ranked CSVs

    output:
    path "*.pt",   emit: model_weights
    path "*.json", emit: model_config
    path "training_history.csv", emit: history

    script:
    """
    mkdir -p rankings
    cp ${ranked_csvs} rankings/
    python ${scripts}/09_train_model.py \
        --config   ${params.config_file} \
        --data-dir rankings              \
        --out-dir  .
    """
}


process VISUALIZE {
    tag "visualize"
    publishDir "${params.outdir}/plots", mode: "copy", overwrite: true

    input:
    path ranked_csvs
    path detailed_csvs

    output:
    path "*.png",                 emit: plots
    path "final_top20_aptamers.csv", emit: final_ranking

    script:
    """
    mkdir -p rankings
    cp ${ranked_csvs}   rankings/ 2>/dev/null || true
    cp ${detailed_csvs} rankings/ 2>/dev/null || true
    python ${scripts}/10_visualize_results.py \
        --config     ${params.config_file} \
        --rank-dir   rankings              \
        --out-dir    .
    """
}


// ════════════════════════════════════════════════════════════════════════════
//  WORKFLOW
//  NOTE: This workflow runs round 0 (initial pool) only via Nextflow.
//  For multi-round SELEX, use run_pipeline.sh which calls 08_selex_iteration.py
//  directly. Implementing full iterative SELEX in Nextflow requires dynamic
//  workflows; the current implementation demonstrates round 0 end-to-end.
// ════════════════════════════════════════════════════════════════════════════

workflow {

    log.info """
    ╔══════════════════════════════════════════════════════╗
    ║       Digital SELEX — TCA Cycle Metabolites          ║
    ╠══════════════════════════════════════════════════════╣
    ║  Rounds:       ${params.rounds}
    ║  Pool size:    ${params.initial_pool_size}
    ║  Exhaustiveness: ${params.exhaustiveness}
    ╚══════════════════════════════════════════════════════╝
    """.stripIndent()

    // Round 0 ──────────────────────────────────────────────────────────────

    // 1. Fetch ligands
    FETCH_LIGANDS()

    // 2. Generate initial sequence pool
    GENERATE_SEQUENCES(0)

    // 3. Predict 2D structures
    PREDICT_2D_STRUCTURE(GENERATE_SEQUENCES.out.fasta, 0)

    // 4. Generate 3D structures
    GENERATE_3D_STRUCTURE(PREDICT_2D_STRUCTURE.out.csv, 0)

    // 5. Prepare docking files
    PREPARE_DOCKING(
        GENERATE_3D_STRUCTURE.out.pdb_files.collect(),
        FETCH_LIGANDS.out.sdf_files.collect(),
        0
    )

    // 6. Run docking
    RUN_DOCKING(
        PREPARE_DOCKING.out.ligand_pdbqts.collect(),
        PREPARE_DOCKING.out.receptor_pdbqts.collect(),
        PREPARE_DOCKING.out.grid_files.collect(),
        0
    )

    // 7. Score aptamers
    SCORE_APTAMERS(
        RUN_DOCKING.out.csv,
        PREDICT_2D_STRUCTURE.out.csv,
        GENERATE_SEQUENCES.out.csv,
        0
    )

    // 8. Train ML model (uses scores from round 0)
    TRAIN_MODEL(
        SCORE_APTAMERS.out.ranked.collect()
    )

    // 9. Visualize
    VISUALIZE(
        SCORE_APTAMERS.out.ranked.collect(),
        SCORE_APTAMERS.out.detailed.collect()
    )

    // Final output message
    VISUALIZE.out.final_ranking.view { file ->
        "\n✓ Pipeline complete. Final aptamer ranking: ${file}"
    }
}
