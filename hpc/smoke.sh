#!/bin/bash
#SBATCH --job-name=t2_smoke
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:2g.20gb:1
#SBATCH --mem=32G
#SBATCH --time=02:00:00
set -eo pipefail
cd "${TASK2_CODE:-${SLURM_SUBMIT_DIR:?Submit from repository root}}"
source hpc/env.sh
hostname
java -version
python -c 'import sys,pyspark,torch; print(sys.executable,pyspark.__version__,pyspark.__file__); print(torch.__version__,torch.cuda.is_available()); assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))'
RUN_SPARK_TESTS=1 python -m pytest tests -q
spark-submit --master 'local[8]' --driver-memory 8g prepare.py --train "${TASK2_RAW}/train.csv" --test "${TASK2_RAW}/test.csv" --out "${TASK2_WORK}/smoke_prepared" --master 'local[8]' --partitions 4 --limit 10000
python train.py --data-dir "${TASK2_WORK}/smoke_prepared" --output-dir "${TASK2_WORK}/smoke_model" --objective forward --device cuda --amp --smoke --max-train-rows 2000 --max-epochs 1 --batch-size 64 --hash-buckets 4096 --embedding-dim 16 --cnn-channels 16 --hidden-dim 32 --workers 0 --threads 2 --evaluate-final
python postprocess.py --cal "${TASK2_WORK}/smoke_model/predictions_cal.csv" --eval "${TASK2_WORK}/smoke_model/predictions_eval.csv" --test "${TASK2_WORK}/smoke_model/predictions_test.csv" --sample "${TASK2_RAW}/sampleSubmission.csv" --metadata "${TASK2_WORK}/smoke_prepared/metadata.json" --out "${TASK2_WORK}/smoke_submission"
echo TASK2_SMOKE_ALL_PASSED
