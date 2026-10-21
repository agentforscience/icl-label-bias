#!/bin/bash
# usage: gpu_run.sh <min_free_MiB> <model> <tag> [gpu_mem_gb]
# Waits until some GPU has >= min_free MiB free, launches run_model.py there,
# retries on failure (waiting does not count as a try). Writes logs/<tag>.status.
cd "$(dirname "$0")/.."
source .venv/bin/activate
export HF_HOME=/data/transformers HF_HUB_OFFLINE=1 VLLM_USE_FLASHINFER_SAMPLER=0
min=$1; model=$2; tag=$3; gb=$4; tries=0
sfx=""; [ "$PROMPTS" = "prompts_supp" ] && sfx="__supp"
[ -f results/raw/$tag$sfx.jsonl ] && { echo DONE > logs/$tag$sfx.status; exit 0; }
while [ $tries -lt 8 ]; do
  gpu=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | sort -t, -k2 -nr | awk -F', ' -v m=$min '$2>=m{print $1; exit}')
  if [ -z "$gpu" ]; then echo "WAIT $(date +%T)" > logs/$tag$sfx.status; sleep 30; continue; fi
  echo "RUN gpu$gpu try$tries $(date +%T)" > logs/$tag$sfx.status
  CUDA_VISIBLE_DEVICES=$gpu python src/run_model.py $model $tag $gb >> logs/$tag$sfx.log 2>&1
  if [ -f results/raw/$tag$sfx.jsonl ]; then echo DONE > logs/$tag$sfx.status; exit 0; fi
  tries=$((tries+1)); sleep 30
done
echo FAILED > logs/$tag$sfx.status
