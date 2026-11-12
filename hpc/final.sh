#!/bin/bash
#SBATCH --job-name=t2_final
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:2g.20gb:1
#SBATCH --mem=32G
#SBATCH --time=02:00:00
set -eo pipefail
cd "${TASK2_CODE:-${SLURM_SUBMIT_DIR:?Submit from repository root}}"
source hpc/env.sh
name="${1:-E4_seed43}"
python train.py --data-dir "${TASK2_WORK}/prepared" --output-dir "${TASK2_WORK}/final_${name}" --predict-only --checkpoint "${TASK2_WORK}/runs/${name}/best.pt" --evaluate-final --device cuda --amp --seed 43 --workers 2 --threads 4
python postprocess.py --cal "${TASK2_WORK}/final_${name}/predictions_cal.csv" --eval "${TASK2_WORK}/final_${name}/predictions_eval.csv" --test "${TASK2_WORK}/final_${name}/predictions_test.csv" --sample "${TASK2_RAW}/sampleSubmission.csv" --metadata "${TASK2_WORK}/prepared/metadata.json" --out "${TASK2_WORK}/submission_${name}"
echo FINAL_EVALUATION_AND_EXPORT_PASSED
