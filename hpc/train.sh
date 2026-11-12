#!/bin/bash
#SBATCH --job-name=t2_train
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:2g.20gb:1
#SBATCH --mem=32G
#SBATCH --time=24:00:00
set -eo pipefail
cd "${TASK2_CODE:-${SLURM_SUBMIT_DIR:?Submit from repository root}}"
source hpc/env.sh
objective="${1:?Usage: sbatch hpc/train.sh OBJECTIVE NAME [extra train args]}"
name="${2:?Missing run name}"
shift 2
python train.py --data-dir "${TASK2_WORK}/prepared" --output-dir "${TASK2_WORK}/runs/${name}" --objective "$objective" --device cuda --amp --max-epochs 3 --patience 2 --batch-size 128 --workers 2 --threads 4 "$@"
