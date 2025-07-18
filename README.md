# MS2CrossCompare Workflow

## Overview

This workflow enables efficient all-vs-all MS2 spectra comparison between two folders, supporting both mzML and mgf formats. It uses fast indexing and cosine similarity, and is designed for large-scale GNPS2 batch spectral comparison tasks.

### Main Features
- Input two folders (each can contain multiple mzML/mgf files); all spectra are automatically discovered recursively.
- Only compares spectra between folder1 and folder2 (no intra-folder comparison).
- Flexible parameter configuration: tolerance, threshold, minimum matched peaks, alignment strategy, peak filtering, threads, and output filename.
- Outputs all high-scoring matches as a TSV file with columns: set1, set2, delta_mz, cosine.

### Key Parameters
| Parameter             | Description                                              | Default Value              |
|----------------------|---------------------------------------------------------|----------------------------|
| inputspectra1        | Path to input folder 1                                  | (must be specified)        |
| inputspectra2        | Path to input folder 2                                  | (must be specified)        |
| tolerance            | MS2 tolerance (Da)                                      | 0.01                       |
| threshold            | Cosine similarity threshold                             | 0.7                        |
| minmatches           | Minimum number of matched peaks                         | 6                          |
| alignment_strategy   | Alignment strategy (index_single_charge/index_multi_charge) | index_single_charge    |
| enable_peak_filtering| Enable peak filtering (yes/no)                          | no                         |
| threads              | Number of threads                                       | 1                          |
| output               | Output filename                                         | cross_compare_results.tsv  |

### Usage
#### Command Line
```bash
python bin/ms2crosscompare.py data/round1 data/round2 \
    --tolerance 0.01 \
    --threshold 0.7 \
    --minmatches 6 \
    --alignment_strategy index_single_charge \
    --enable_peak_filtering no \
    --threads 1 \
    --output cross_compare_results.tsv
```

#### Nextflow Workflow
All parameters can be configured via workflowinput.yaml and are compatible with GNPS2 web interface forms.
```bash
nextflow run nf_workflow.nf --inputspectra1 data/round1 --inputspectra2 data/round2 --tolerance 0.01 --threshold 0.7 --minmatches 6 --alignment_strategy index_single_charge --enable_peak_filtering no --threads 1 --output cross_compare_results.tsv
```

### Output
The output is a TSV file with the following columns:
- set1: Spectrum ID from folder 1 (including filename and scan number)
- set2: Spectrum ID from folder 2
- delta_mz: Precursor mass difference
- cosine: Cosine similarity score

---

## Workflow Download and Test
Make sure to have the NextflowModule updated if you plan on using it:

```
git submodule update --init --remote --recursive
```

To run the workflow to test simply do

```
make run
```

To learn NextFlow checkout this documentation:

https://www.nextflow.io/docs/latest/index.html

## Installation

You will need to have conda, mamba, and nextflow installed to run things locally. 

## GNPS2 Workflow Input information

Check the definition for the workflow input and display parameters:
https://wang-bioinformatics-lab.github.io/GNPS2_Documentation/workflowdev/

## Deployment to GNPS2

In order to deploy, we have a set of deployment tools that will enable deployment to the various gnps2 systems. To run the deployment, you will need the following setup steps completed:

1. Checked out of the deployment submodules
1. Conda environment and dependencies
1. SSH configuration updated

### Checking out the deployment submodules

use the following commands from the deploy_gnps2 folder. 

You might need to checkout the module, do this by running

```
git submodule init
git submodule update
```

You will also need to specify the user on the server that you've been given that your public key has been associated with. If you want to not enter this every time you do a deployment, you can create a Makefile.credentials file in the deploy_gnps2 folder with the following contents

```
USERNAME=<enter the username>
```

### Deployment Dependencies

You will need to install the dependencies in GNPS2_DeploymentTooling/requirements.txt on your own local machine. 

You can find this [here](https://github.com/Wang-Bioinformatics-Lab/GNPS2_DeploymentTooling).

One way to do this is to use conda to create an environment, for example:

```
conda create -n deploy python=3.8
pip install -r GNPS2_DeploymentTooling/requirements.txt
```

### SSH Configuration

Also update your ssh config file to include the following ssh target:

```
Host ucr-gnps2-dev
    Hostname ucr-lemon.duckdns.org
```

### Deploying to Dev Server

To deploy to development, use the following command, if you don't have your ssh public key installed onto the server, you will not be able to deploy.

```
make deploy-dev
```

### Deploying to Production Server

To deploy to production, use the following command, if you don't have your ssh public key installed onto the server, you will not be able to deploy.

```
make deploy-prod
```

