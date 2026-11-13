#!/bin/bash
#SBATCH --job-name=t2_text_repr
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:2g.20gb:1
#SBATCH --mem=32G
#SBATCH --time=08:00:00
set -eo pipefail
cd "${TASK2_CODE:-${SLURM_SUBMIT_DIR:?Submit from repository root}}"
source hpc/env.sh
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
python -m pytest tests -q
for condition in 'separate mean' 'joint mean_max' 'separate mean_max'; do
    read -r layout pooling <<< "$condition"
    python train.py --data-dir "$TASK2_WORK/smoke_prepared" --output-dir "$TASK2_WORK/text_smoke_${layout}_${pooling}" --objective forward --metadata --text-layout "$layout" --pooling "$pooling" --device cuda --amp --smoke --max-train-rows 128 --max-epochs 1 --patience 1 --batch-size 64 --hash-buckets 4096 --embedding-dim 16 --cnn-channels 16 --hidden-dim 32 --workers 0 --threads 2
 done
for condition in 'T1_separate_mean_seed43 separate mean' 'T2_joint_meanmax_seed43 joint mean_max' 'T3_separate_meanmax_seed43 separate mean_max'; do
    read -r name layout pooling <<< "$condition"
    python train.py --data-dir "$TASK2_WORK/prepared" --output-dir "$TASK2_WORK/runs/$name" --objective forward --metadata --text-layout "$layout" --pooling "$pooling" --device cuda --amp --seed 43 --max-epochs 4 --patience 4 --batch-size 128 --workers 2 --threads 4 --lr-decay-start 4 --lr-decay-every 2 --lr-decay-factor 0.5 --save-every-epoch
    python train.py --data-dir "$TASK2_WORK/prepared" --output-dir "$TASK2_WORK/runs/${name}_latent" --predict-only --checkpoint "$TASK2_WORK/runs/$name/best_latent.pt" --device cuda --amp --seed 43 --workers 2 --threads 4
 done
python run_text_ablation.py --stage cal --work "$TASK2_WORK" --out "$TASK2_WORK/text_delivery_20261113" --sample "$TASK2_RAW/sampleSubmission.csv"
echo TEXT_ABLATION_TRAINING_AND_CAL_SELECTION_COMPLETE
python run_text_ablation.py --stage eval --work "$TASK2_WORK" --out "$TASK2_WORK/text_delivery_20261113"
echo TEXT_ABLATION_FULLY_COMPLETE
