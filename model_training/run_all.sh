#!/bin/bash
# Full pipeline on Ada (IIIT HPC). Runs the four cases as parallel processes so
# they share the one allocated GPU (XGBoost, XGBoost-RF and the MLP all use CUDA).
#
#   Batch job:            sbatch run_all.sh
#   Inside an allocation: srun --jobid=<JOBID> --overlap bash run_all.sh
#   One model at a time:  MODELS=xgboost STEPS=train sbatch run_all.sh
#                         (submit_by_model.sh chains one such job per family)
#
#SBATCH -J leakage-ml
#SBATCH -A research
#SBATCH -p u22
#SBATCH --qos=low
#SBATCH --constraint=2080ti
#SBATCH --gres=gpu:1
#SBATCH -c 10
#SBATCH --mem-per-cpu=2G
#SBATCH --time=12:00:00
#SBATCH -o logs/run_all_%j.out

set -euo pipefail
# sbatch runs a spooled copy of this script, so $0 is not in model_training/.
cd "${SLURM_SUBMIT_DIR:-$(dirname "$(readlink -f "$0")")}"
mkdir -p logs
# CLEAN=1: start from empty outputs; old logs are kept in logs/prev/ for post-mortems.
# (stale .nfs* files held open elsewhere cannot be removed; ignore them)
if [ "${CLEAN:-0}" = 1 ]; then
  mkdir -p logs/prev && mv logs/*.log logs/prev/ 2>/dev/null || true
  rm -rf results saved_models 2>/dev/null || true
fi
module load u22/python/3.12.4 2>/dev/null || true
source "${VENV:-$HOME/DVD_project/.venv}/bin/activate"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
export PYTHONUNBUFFERED=1          # live log files
export REQUIRE_GPU=${REQUIRE_GPU:-1}   # abort instead of silently training on CPU
python -c "import torch, xgboost; assert torch.cuda.is_available(); print('torch', torch.__version__, 'xgboost', xgboost.__version__, 'CUDA OK')"

EXTRA=${EXTRA:-}            # e.g. EXTRA=--quick for a smoke test
JOBS_PER_CASE=${JOBS_PER_CASE:-2}
MODELS=${MODELS:-"poly_elasticnet random_forest xgboost mlp"}   # families for step 1
STEPS=${STEPS:-"train raw eval size"}                            # which steps to run
TAG=${MODELS// /+}
has_step() { [[ " $STEPS " == *" $1 "* ]]; }

# Runs train.py for the four cases as parallel processes; fails if any case fails.
train_cases() {   # $1 = log prefix, rest = train.py args
  local prefix=$1; shift
  local pids=()
  for case in NMOS_ON NMOS_OFF PMOS_ON PMOS_OFF; do
    python train.py --cases "$case" --n-jobs "$JOBS_PER_CASE" $EXTRA "$@" > "logs/${prefix}_${case}.log" 2>&1 &
    pids+=($!)
  done
  for p in "${pids[@]}"; do wait "$p"; done
}

# 1. Tune + fit the chosen families for each case (physics features).
if has_step train; then train_cases "train_${TAG}" --models $MODELS; fi

# 2. Feature ablation: same pipeline on raw inputs only (XGBoost + MLP).
if has_step raw; then train_cases "train_raw" --models xgboost mlp --feature-set raw; fi

# 3. Compare on validation, choose per case, evaluate once on test, plots.
if has_step eval; then python evaluate.py > logs/evaluate.log 2>&1; fi

# 4. Training-set-size learning curve for the chosen models.
if has_step size; then python data_size_curve.py > logs/data_size_curve.log 2>&1; fi

echo "done ($STEPS): see results/ and saved_models/"
