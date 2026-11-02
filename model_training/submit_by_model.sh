#!/bin/bash
# Submit the pipeline as one Slurm job per model family instead of one long job.
# Results for each family land in results/candidates/ as soon as its job ends,
# and a failed family does not block the others (afterany dependencies). QoS
# "low" allows one GPU at a time, so the jobs run one after another; the
# fastest families go first.
#
#   bash submit_by_model.sh              # start now (first job clears old outputs)
#   bash submit_by_model.sh <JOBID>      # start after job <JOBID> ends
#
# Run on the Ada login node from model_training/.
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")"
mkdir -p logs

dep=${1:+--dependency=afterany:$1}
clean=1
submit() {   # $1 = job name, $2 = time limit, rest = VAR=value settings for run_all.sh
  local name=$1 time=$2; shift 2
  local id
  id=$(env CLEAN=$clean "$@" sbatch --parsable -J "$name" --time="$time" -o "logs/%x_%j.out" $dep run_all.sh)
  echo "$name: job $id"
  dep="--dependency=afterany:$id"
  clean=0
}

submit leak-enet  02:00:00 MODELS=poly_elasticnet STEPS=train
submit leak-mlp   04:00:00 MODELS=mlp             STEPS=train
submit leak-rf    04:00:00 MODELS=random_forest   STEPS=train
submit leak-xgb   10:00:00 MODELS=xgboost         STEPS=train
# QoS allows 5 submitted jobs per user, so the ablation, evaluation and
# data-size curve share the last job.
submit leak-final 14:00:00 STEPS="raw eval size"
squeue -u "$USER"
