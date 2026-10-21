# ICL Label Bias: When Demonstrations Teach the Wrong Lesson

This is a controlled, fully paired study of how four spurious statistics in few-shot demonstrations shift LLM predictions toward a target label:
- label-frequency skew
- recency
- label position (primacy)
- label-length correlation

The setup is 4 NLU tasks (SST-2, RTE, AG News, MRPC) × 8 local models (Qwen2.5-Instruct 0.5B–14B, Qwen2.5-1.5B base, Llama-3.2-1B, Llama-3.1-8B), with 80k scored prompts.

## Key findings
- **Every bias is much smaller than hypothesised.** Frequency skew (80/20) shifts predictions by +4 to +5 pp in small Qwen-Instruct models, +8.9 pp in a 1.5B base model, and +16.9 pp in Llama-3.2-1B. It shrinks to about 0 at 8–14B (slope −3.7 pp per 10× parameters).
- **Recency (last 2 demos = T) gives no 8–15 pp pull.** At ≥3B it becomes a small, significant *anti*-recency effect (−1.7 to −3.6 pp).
- **Label-length correlation has no reliable effect** at any scale, and no model learns a length → label rule.
- **Calibration (ECE) does not reliably worsen** under any bias (1/32 bootstrap CIs barely above zero, uncorrected).
- **Bias concentrates in ambiguous tasks** (MRPC: up to +42 pp) and **extreme doses** in small models (100% one label: +16 to +36 pp at ≤1.5B, only +3 pp at 7B). For models ≥3B, plain demo-set resampling flips more predictions (4–23%) than any manipulation moves them.

Full details are in [REPORT.md](REPORT.md). The design is in [planning.md](planning.md).

## Reproduce
```bash
uv venv --python 3.12 && source .venv/bin/activate && uv sync   # vllm>=0.10, pandas, scipy, matplotlib
export HF_HOME=/data/transformers HF_HUB_OFFLINE=1 VLLM_USE_FLASHINFER_SAMPLER=0
# RTE/MRPC parquet files: datasets/raw/{rte,mrpc}/{train,validation}.parquet (from hf nyu-mll/glue)
python src/build_prompts.py            # 10k main prompts
python src/build_prompts.py supp       # 6k dose-response prompts
CUDA_VISIBLE_DEVICES=0 python src/run_model.py Qwen/Qwen2.5-0.5B-Instruct qwen0.5b
PROMPTS=prompts_supp python src/run_model.py Qwen/Qwen2.5-0.5B-Instruct qwen0.5b
# (or src/gpu_run_v2.sh <min_free_MiB> <hf_id> <tag> [gb] to wait for free GPU memory)
cd src && python analyze.py && python analyze_supp.py
```
The model tags expected by `analyze.py` are: qwen0.5b, qwen1.5b, qwen3b, qwen7b-awq, qwen14b, qwen1.5b-base, llama1b, llama8b-awq.

## Layout
```
src/build_prompts.py   controlled demo construction (5 conditions + 3 dose conditions)
src/run_model.py       vLLM scoring: label-restricted next-token probabilities
src/gpu_run*.sh        wait-for-GPU / retry wrappers
src/analyze.py         shifts, bootstrap CIs, permutation tests, scale slopes, ECE, figures 1-3
src/analyze_supp.py    dose-response table + figure 4
datasets/prompts/      all prompts (shared across models)
results/raw/           per-model per-prompt label probabilities
results/*.csv          summary tables; results/analysis_output.txt = printed analysis
figures/               fig1_target_shift, fig2_scale, fig3_calibration_accuracy, fig4_dose_response
logs/                  run logs
```
