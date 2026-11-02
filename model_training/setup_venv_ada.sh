#!/bin/bash
# Build the project venv on Ada. Run it ON A COMPUTE NODE (the login gateway has a
# different OS), e.g. inside:
#   srun --pty -p u22 -A research --qos=low --constraint=2080ti --gres=gpu:1 -c 10 --mem-per-cpu=2G bash -l
#   bash setup_venv_ada.sh
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")"
VENV=${VENV:-~/DVD_project/.venv}

/usr/local/apps/python-3.12.4/bin/python3 -m venv "$VENV"    # = module load u22/python/3.12.4
. "$VENV/bin/activate"
pip install --no-cache-dir --upgrade pip
pip install --no-cache-dir -r requirements.txt

# torch and xgboost-cu12 both ship nvidia-nccl-cu12 into the same folder; make sure
# the copy torch was built against is the one installed (else: undefined symbol
# ncclCommGrow on nodes whose system NCCL is older).
pip install --no-cache-dir --force-reinstall --no-deps nvidia-nccl-cu12==2.29.3

python -c "import sklearn, xgboost, torch; assert torch.cuda.is_available(); \
print('OK sklearn', sklearn.__version__, 'xgboost', xgboost.__version__, 'torch', torch.__version__, torch.cuda.get_device_name(0))"
