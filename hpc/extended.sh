#!/bin/bash
#SBATCH --job-name=t2_extended
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:2g.20gb:1
#SBATCH --mem=32G
#SBATCH --time=06:00:00
set -eo pipefail
cd "${TASK2_CODE:-${SLURM_SUBMIT_DIR:?Submit from repository root}}"
source hpc/env.sh
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
python -m pytest tests -q
python train.py --data-dir "$TASK2_WORK/smoke_prepared" --output-dir "$TASK2_WORK/extended_smoke" --objective forward --metadata --device cuda --amp --smoke --max-train-rows 128 --max-epochs 2 --patience 2 --batch-size 64 --hash-buckets 4096 --embedding-dim 16 --cnn-channels 16 --hidden-dim 32 --workers 0 --threads 2 --lr-decay-start 2 --save-every-epoch
python ensemble_predictions.py --runs-root "$TASK2_WORK/runs" --models E4 E4_seed43 --name E4_two_seed --metadata "$TASK2_WORK/prepared/metadata.json" --sample "$TASK2_RAW/sampleSubmission.csv" --out "$TASK2_WORK/ensemble_E4_two_seed"
python ensemble_predictions.py --runs-root "$TASK2_WORK/runs" --models E3 E4 E3_seed43 E4_seed43 --name E34_four_model --metadata "$TASK2_WORK/prepared/metadata.json" --sample "$TASK2_RAW/sampleSubmission.csv" --out "$TASK2_WORK/ensemble_E34_four_model"
echo EXISTING_ENSEMBLES_COMPLETED
python train.py --data-dir "$TASK2_WORK/prepared" --output-dir "$TASK2_WORK/runs/E4_long8_seed43" --objective forward --metadata --device cuda --amp --seed 43 --max-epochs 8 --patience 8 --batch-size 128 --workers 2 --threads 4 --lr-decay-start 4 --lr-decay-every 2 --lr-decay-factor 0.5 --save-every-epoch
python train.py --data-dir "$TASK2_WORK/prepared" --output-dir "$TASK2_WORK/runs/E4_long8_seed43_latent" --predict-only --checkpoint "$TASK2_WORK/runs/E4_long8_seed43/best_latent.pt" --device cuda --amp --seed 43 --workers 2 --threads 4
python ensemble_predictions.py --runs-root "$TASK2_WORK/runs" --models E4_long8_seed43_latent --name E4_long8_single --metadata "$TASK2_WORK/prepared/metadata.json" --sample "$TASK2_RAW/sampleSubmission.csv" --out "$TASK2_WORK/submission_E4_long8_single"
python ensemble_predictions.py --runs-root "$TASK2_WORK/runs" --models E4 E4_long8_seed43_latent --name E4_long8_plus_seed42 --metadata "$TASK2_WORK/prepared/metadata.json" --sample "$TASK2_RAW/sampleSubmission.csv" --out "$TASK2_WORK/ensemble_E4_long8_plus_seed42"
echo EXTENDED_TRAINING_AND_EXPORT_COMPLETED
