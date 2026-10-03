#!/bin/bash
#SBATCH --job-name=cert_all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32gb
#SBATCH --time=5:00:00
#SBATCH --gpus=h100:1
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.log
module purge
eval "$(micromamba shell hook --shell bash)"
micromamba activate fenics-env
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"   # run from the repo root: sbatch jobs/<script>.sh
export MPLBACKEND=Agg
python -u src/certify_all_pores.py
