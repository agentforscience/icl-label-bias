# Data
- SST-2 (`stanfordnlp/sst2`) and AG News (`fancyzhx/ag_news`) are read from the local HF cache (`/data/transformers/hub`).
- RTE and MRPC: `raw/{rte,mrpc}/{train,validation}.parquet`, downloaded from
  `https://huggingface.co/datasets/nyu-mll/glue/resolve/main/<task>/<split>-00000-of-00001.parquet`.
- `prompts/prompts.jsonl` (10,000) and `prompts/prompts_supp.jsonl` (6,000) are generated deterministically by `src/build_prompts.py`.
