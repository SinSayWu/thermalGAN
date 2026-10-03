#!/bin/bash
#SBATCH --job-name=sg08_p
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64gb
#SBATCH --time=48:00:00
#SBATCH --gpus=h100:1
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.log
module purge
eval "$(micromamba shell hook --shell bash)"
micromamba activate fenics-env
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"   # run from the repo root: sbatch jobs/<script>.sh
export MPLBACKEND=Agg
: "${PORES:=8}"; export PORES   # pore count: sbatch --export=ALL,PORES=4 jobs/jobs_smoothgen08_pores.sh
echo "[job] host=$(hostname) JOB=$SLURM_JOB_ID PORES=$PORES"
# Winning smooth-border arch (sigma=0.8) applied to $PORES-pore dataset.
python -u src/ResDCGAN_GP_thermal_strongD_smoothgen08.py
