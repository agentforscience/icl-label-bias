"""Score every prompt with one local model via vLLM.

For each prompt we take the next-token distribution restricted to the label
words' first tokens (leading space), renormalised over the label set
(logprobs_mode="processed_logprobs" + allowed_token_ids).  Greedy prediction
= argmax of that distribution.  Output: results/raw/<tag>.jsonl with the
label probability vector per prompt.

usage: python src/run_model.py <hf_model_id> <tag> [gpu_mem_gb]
"""
import os, sys, json, math, time
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
import torch
from vllm import LLM, SamplingParams

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    model, tag = sys.argv[1], sys.argv[2]
    want_gb = float(sys.argv[3]) if len(sys.argv) > 3 else None
    pfile = os.environ.get("PROMPTS", "prompts")  # "prompts" (main) or "prompts_supp"
    out_path = f"{ROOT}/results/raw/{tag}{'' if pfile == 'prompts' else '__supp'}.jsonl"
    rows = [json.loads(l) for l in open(f"{ROOT}/datasets/prompts/{pfile}.jsonl")]
    free, total = torch.cuda.mem_get_info()
    budget = (want_gb * 1e9) if want_gb else free - 1.5e9
    budget = min(budget, free - 1.2e9)
    util = budget / total
    print(f"[{tag}] free={free/1e9:.1f}GB util={util:.3f}", flush=True)
    small = budget < 12e9
    llm = LLM(model=model, gpu_memory_utilization=util, max_model_len=3072,
              enable_prefix_caching=True, enforce_eager=small, seed=0,
              max_num_seqs=32 if small else 128,
              max_num_batched_tokens=2048 if small else 8192,
              logprobs_mode="processed_logprobs")
    tok = llm.get_tokenizer()
    t0 = time.time()
    with open(out_path + ".partial", "w") as f:
        for task in sorted({r["task"] for r in rows}):
            sub = [r for r in rows if r["task"] == task]
            labels = sub[0]["labels"]
            ids = [tok.encode(" " + w, add_special_tokens=False) for w in labels]
            assert all(len(i) >= 1 for i in ids)
            first = [i[0] for i in ids]
            assert len(set(first)) == len(first), (task, ids)
            sp = SamplingParams(max_tokens=1, temperature=0.0, logprobs=len(labels),
                                allowed_token_ids=first)
            outs = llm.generate([r["prompt"] for r in sub], sp, use_tqdm=False)
            for r, o in zip(sub, outs):
                lp = o.outputs[0].logprobs[0]
                logp = [lp[i].logprob if i in lp else -math.inf for i in first]
                m = max(logp); p = [math.exp(x - m) for x in logp]; s = sum(p)
                p = [x / s for x in p]
                f.write(json.dumps(dict(task=task, test_id=r["test_id"], T=r["T"], cond=r["cond"],
                                        gold=r["gold"], test_nwords=r["test_nwords"], probs=p)) + "\n")
            print(f"[{tag}] {task} done {time.time()-t0:.0f}s", flush=True)
    os.replace(out_path + ".partial", out_path)
    print(f"[{tag}] DONE {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
