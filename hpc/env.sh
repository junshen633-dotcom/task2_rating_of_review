#!/bin/bash
# Source from SLURM scripts after changing to the submitted repository.
module load anaconda3
source /cm/shared/apps/anaconda3/etc/profile.d/conda.sh
conda activate "${TASK2_ENV:-task2_v2}"
export PYTHONNOUSERSITE=1
unset PYTHONPATH SPARK_HOME SPARK_CONF_DIR PYSPARK_SUBMIT_ARGS JAVA_TOOL_OPTIONS _JAVA_OPTIONS JDK_JAVA_OPTIONS
export JAVA_HOME="$CONDA_PREFIX"
export PATH="$JAVA_HOME/bin:$PATH"
export PYSPARK_PYTHON="$CONDA_PREFIX/bin/python"
export PYSPARK_DRIVER_PYTHON="$CONDA_PREFIX/bin/python"
export SPARK_LOCAL_IP=127.0.0.1
export TASK2_RAW="${TASK2_RAW:-/scratch/$USER/data}"
export TASK2_WORK="${TASK2_WORK:-/scratch/$USER/task2_v2}"
export SPARK_LOCAL_DIRS="$TASK2_WORK/spark_tmp/${SLURM_JOB_ID:-local}"
mkdir -p "$SPARK_LOCAL_DIRS"
cd "${TASK2_CODE:-${SLURM_SUBMIT_DIR:?Submit from repository root}}"
