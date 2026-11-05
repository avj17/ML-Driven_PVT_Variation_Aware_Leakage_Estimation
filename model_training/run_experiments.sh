#!/bin/bash
#SBATCH -J leak-exp
#SBATCH -A research
#SBATCH -p u22
#SBATCH --qos=low
#SBATCH --constraint=2080ti
#SBATCH --gres=gpu:1
#SBATCH -c 10
#SBATCH --mem-per-cpu=2G
#SBATCH --time=06:00:00
#SBATCH -o logs/%x_%j.out
# MLP recipe study (mlp_experiments.py): the 4 cases in parallel on one GPU.
#   sbatch run_experiments.sh            QUICK=1 sbatch run_experiments.sh  (smoke test)
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$(readlink -f "$0")")}"
mkdir -p logs
. ~/DVD_project/.venv/bin/activate
export PYTHONUNBUFFERED=1 REQUIRE_GPU=1
python -c "import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))"

EXTRA=""; [ "${QUICK:-0}" = 1 ] && EXTRA="--quick"
pids=()
for c in NMOS_ON NMOS_OFF PMOS_ON PMOS_OFF; do
  python mlp_experiments.py --cases "$c" $EXTRA > "logs/exp_${c}.log" 2>&1 &
  pids+=($!)
done
rc=0
for p in "${pids[@]}"; do wait "$p" || rc=1; done
exit $rc
