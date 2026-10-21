"""Dose-response analysis: skew 60/80/100% and recency last-2 vs last-4,
each paired with the main run's clean prompt for the same (task, item, T)."""
import os, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from analyze import MODELS, RES, FIG, TASKS, load, boot_task_mean

def main():
    d = load()
    rows = []
    for tag, name, size, fam in MODELS:
        p = f"{RES}/raw/{tag}__supp.jsonl"
        if not os.path.exists(p) or name not in d.model.unique():
            continue
        s = pd.read_json(p, lines=True)
        s["pred"] = s.probs.apply(np.argmax); s["predT"] = (s.pred == s["T"]).astype(float)
        main_ = d[d.model == name][["task", "test_id", "T", "cond", "predT"]]
        allc = pd.concat([main_, s[["task", "test_id", "T", "cond", "predT"]]])
        w = allc.pivot_table(index=["task", "test_id", "T"], columns="cond", values="predT").reset_index()
        for c in ["skew60", "skew", "skew100", "recency", "recency4"]:
            w[c + "_d"] = w[c] - w["clean"]
            it = w.groupby(["task", "test_id"])[c + "_d"].mean().reset_index()
            pt, lo, hi, _ = boot_task_mean(it, c + "_d")
            per = it.groupby("task")[c + "_d"].mean()
            rows.append(dict(model=name, cond=c, delta_pp=100 * pt, lo=100 * lo, hi=100 * hi,
                             **{f"delta_{t}": 100 * per[t] for t in TASKS}))
    r = pd.DataFrame(rows)
    r.to_csv(f"{RES}/dose_response.csv", index=False)
    print(r.round(1).to_string(index=False))
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    for m, g in r.groupby("model", sort=False):
        g = g.set_index("cond")
        sk = g.loc[["skew60", "skew", "skew100"]]
        ax[0].errorbar([60, 80, 100], sk.delta_pp, yerr=[sk.delta_pp - sk.lo, sk.hi - sk.delta_pp], marker="o", capsize=3, label=m)
        rc = g.loc[["recency", "recency4"]]
        ax[1].errorbar([2, 4], rc.delta_pp, yerr=[rc.delta_pp - rc.lo, rc.hi - rc.delta_pp], marker="o", capsize=3, label=m)
    ax[0].set_xlabel("% of demos with target label"); ax[1].set_xlabel("# final demos with target label")
    ax[1].set_xticks([2, 4])
    for a in ax: a.axhline(0, color="k", lw=.5); a.set_ylabel("Δ target rate vs clean (pp)")
    ax[0].set_title("Frequency-skew dose-response"); ax[1].set_title("Recency dose-response")
    ax[0].legend(fontsize=7, frameon=False)
    fig.tight_layout(); fig.savefig(f"{FIG}/fig4_dose_response.png", dpi=150); plt.close(fig)

if __name__ == "__main__":
    main()
