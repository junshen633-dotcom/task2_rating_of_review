#!/bin/bash
#SBATCH --job-name=t2_prepare
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
set -eo pipefail
cd "${TASK2_CODE:-${SLURM_SUBMIT_DIR:?Submit from repository root}}"
source hpc/env.sh
spark-submit --master local[8] --driver-memory 16g prepare.py --train "${TASK2_RAW}/train.csv" --test "${TASK2_RAW}/test.csv" --out "${TASK2_WORK}/prepared" --master local[8] --partitions 64
