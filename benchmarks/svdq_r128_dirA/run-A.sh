#!/usr/bin/env bash
# usage: run-A.sh <gpu> <nsys-tag|-> <python script and args...>   (direction A only)
set -euo pipefail
gpu=$1; tag=$2; shift 2
W=$WORKSPACE; R=$W/svdq-rank128-20261019
exec 9>"$W/.svdq-$(hostname)-gpu${gpu}.lock"; flock -n 9 || { echo busy; exit 75; }
[[ -z "$(nvidia-smi -i $gpu --query-compute-apps=pid --format=csv,noheader)" ]] || { echo occupied; exit 75; }
mkdir -p $R/profiles/dirA $R/tmp
export ENROOT_DATA_PATH=$W/enroot/data ENROOT_RUNTIME_PATH=$W/enroot/runtime
export CUDA_VISIBLE_DEVICES=$(nvidia-smi -i $gpu --query-gpu=uuid --format=csv,noheader)
if [[ "$tag" == "-" ]]; then pre=""; else pre="nsys profile -t cuda,nvtx --cuda-graph-trace=node -o \$R/profiles/dirA/$tag -f true"; fi
enroot start --root --mount $W:/workspace --env CUDA_VISIBLE_DEVICES --env SVDQ_A_OVERLAP svdq-ctk132-screen bash -c "
export PATH=/workspace/fi-svdq-perf-20260914/venv/bin:\$PATH
R=/workspace/svdq-rank128-20261019; cd \$R/repo-dirA
export PYTHONPATH=\$R/repo-dirA:\$R/repo-dirA/benchmarks FLASHINFER_WORKSPACE_BASE=\$R/cache-dirA TMPDIR=\$R/tmp FLASHINFER_CUDA_ARCH_LIST=12.0a MAX_JOBS=16 PYTHONUNBUFFERED=1
$pre python $*; echo run-exit=\$?"
