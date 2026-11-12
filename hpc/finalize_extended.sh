#!/bin/bash
#SBATCH --job-name=t2_ext_final
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:2g.20gb:1
#SBATCH --mem=32G
#SBATCH --time=01:00:00
set -eo pipefail
cd "${TASK2_CODE:-${SLURM_SUBMIT_DIR:?Submit from repository root}}"
source hpc/env.sh
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
python collect_extended.py --work "$TASK2_WORK" --sample "$TASK2_RAW/sampleSubmission.csv" --out "$TASK2_WORK/extended_delivery_20261113"
python evaluate_selected.py --work "$TASK2_WORK" --bundle "$TASK2_WORK/extended_delivery_20261113"
echo EXTENDED_FINAL_REVIEW_COMPLETED
