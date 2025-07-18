#!/usr/bin/env nextflow
nextflow.enable.dsl=2

params.inputspectra1 = "./data/round1"
params.inputspectra2 = "./data/round2"
params.tolerance = 0.01
params.threshold = 0.7
params.threads = 1
params.alignment_strategy = "index_single_charge"
params.enable_peak_filtering = "no" 
params.output = "cross_compare_results.tsv"
params.publishdir = "$launchDir"
params.minmatches = 6
TOOL_FOLDER = "$moduleDir/bin"
MODULES_FOLDER = "$TOOL_FOLDER/NextflowModules"

// GNPS2 Boiler Plate
params.task = "" // This is the GNPS2 task if it is necessary

// COMPATIBILITY NOTE: The following might be necessary if this workflow is being deployed in a slightly different environemnt
// checking if outdir is defined,
// if so, then set publishdir to outdir
if (params.outdir) {
    _publishdir = params.outdir
}
else{
    _publishdir = params.publishdir
}

// Augmenting with nf_output
_publishdir = "${_publishdir}/nf_output"

process ms2CrossCompare {
    publishDir "$_publishdir", mode: 'copy'
    conda "$TOOL_FOLDER/conda_env.yml"

    input:
    path input_folder1
    path input_folder2
    val tolerance
    val threshold
    val threads
    val alignment_strategy
    val enable_peak_filtering
    val minmatches
    val output_file

    output:
    file "${output_file}"

    script:
    """
    python $TOOL_FOLDER/ms2crosscompare.py $input_folder1 $input_folder2 \
        --tolerance $tolerance \
        --threshold $threshold \
        --threads $threads \
        --alignment_strategy $alignment_strategy \
        --enable_peak_filtering $enable_peak_filtering \
        --minmatches $minmatches \
        --output $output_file
    """
}

workflow Main {
    take:
    input_map

    main:
    cross_compare_ch = ms2CrossCompare(
        input_map.inputspectra1,
        input_map.inputspectra2,
        input_map.tolerance,
        input_map.threshold,
        input_map.threads,
        input_map.alignment_strategy,
        input_map.enable_peak_filtering,
        input_map.minmatches,
        input_map.output
    )

    emit:
    cross_compare_ch
}

workflow {
    /* 
    The input map is created to reduce the dependency of the other workflows to the `params`
    */
    input_spectra_ch1 = Channel.fromPath(params.inputspectra1)
    input_spectra_ch2 = Channel.fromPath(params.inputspectra2)
    input_map = [
        inputspectra1: input_spectra_ch1,
        inputspectra2: input_spectra_ch2,
        tolerance: params.tolerance,
        threshold: params.threshold,
        threads: params.threads,
        alignment_strategy: params.alignment_strategy,
        enable_peak_filtering: params.enable_peak_filtering,
        output: params.output,
        minmatches: params.minmatches
    ]
    out = Main(input_map)
    out.view()
}
