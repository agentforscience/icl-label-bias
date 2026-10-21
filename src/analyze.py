"""Analysis of ICL label-bias runs: shifts, accuracy, calibration, stats, figures.

Primary metric: target shift  Δ = P(pred = T | cond) − P(pred = T | clean),
paired on (task, test item, target label T).  Δ is averaged over T and items
within a task, then tasks are weighted equally.  CIs: cluster bootstrap over
test items (resampled within task).  Significance: sign-flip permutation test
on per-item Δ, Holm-corrected over all (model, condition) tests.
"""
import os, json, glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES, FIG = f"{ROOT}/results", f"{ROOT}/figures"
B = 2000
rng = np.random.default_rng(0)
CONDS = ["skew", "recency", "primacy", "length"]
MODELS = [  # tag, label, params (B), family
    ("qwen0.5b", "Qwen2.5-0.5B", 0.5, "qwen"), ("qwen1.5b", "Qwen2.5-1.5B", 1.5, "qwen"),
    ("qwen3b", "Qwen2.5-3B", 3, "qwen"), ("qwen7b-awq", "Qwen2.5-7B-AWQ", 7, "qwen"),
    ("qwen14b", "Qwen2.5-14B", 14, "qwen"), ("qwen1.5b-base", "Qwen2.5-1.5B-base", 1.5, "qwen-base"), ("llama1b", "Llama-3.2-1B", 1.2, "llama"),
    ("llama8b-awq", "Llama-3.1-8B-AWQ", 8, "llama")]
TASKS = ["sst2", "rte", "agnews", "mrpc"]


def load():
    dfs = []
    for tag, name, size, fam in MODELS:
        p = f"{RES}/raw/{tag}.jsonl"
        if not os.path.exists(p):
            continue
        d = pd.read_json(p, lines=True)
        d["model"], d["size"], d["family"] = name, size, fam
        dfs.append(d)
    d = pd.concat(dfs, ignore_index=True)
    P = np.stack(d.probs.apply(lambda x: np.pad(x, (0, 4 - len(x)))).values)
    d["nl"] = d.probs.apply(len)
    d["pred"] = P.argmax(1)
    d["conf"] = P.max(1)
    d["acc"] = (d.pred == d.gold).astype(float)
    d["predT"] = (d.pred == d["T"]).astype(float)
    d["pT"] = P[np.arange(len(d)), d["T"].values]
    with np.errstate(divide="ignore", invalid="ignore"):
        ent = -np.nansum(np.where(P > 0, P * np.log(P), 0), 1)
    d["entropy"] = ent / np.log(d.nl)
    med = d.groupby("task").test_nwords.transform("median")
    d["long_test"] = d.test_nwords > med
    return d


def ece(conf, correct, nb=10):
    bins = np.minimum((conf * nb).astype(int), nb - 1)
    e = 0.0
    for b in range(nb):
        m = bins == b
        if m.any():
            e += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return e


def item_deltas(d, metric="predT"):
    """Per (model, task, test_id, cond) Δ averaged over T, relative to clean."""
    k = ["model", "task", "test_id", "T"]
    w = d.pivot_table(index=k, columns="cond", values=metric).reset_index()
    for c in CONDS:
        w[c] = w[c] - w["clean"]
    return w.groupby(["model", "task", "test_id"])[CONDS].mean().reset_index()


def boot_task_mean(item_df, col, B=B):
    """Equal-task-weighted mean with cluster bootstrap over items within task."""
    groups = [g[col].values for _, g in item_df.groupby("task")]
    point = np.mean([g.mean() for g in groups])
    bs = np.zeros(B)
    for b in range(B):
        bs[b] = np.mean([g[rng.integers(0, len(g), len(g))].mean() for g in groups])
    return point, np.percentile(bs, 2.5), np.percentile(bs, 97.5), bs


def signflip_p(x, n=10000):
    x = np.asarray(x); obs = abs(x.mean())
    s = rng.choice([-1, 1], size=(n, len(x)))
    return (np.sum(np.abs((s * x).mean(1)) >= obs) + 1) / (n + 1)


def holm(ps):
    ps = np.asarray(ps); o = np.argsort(ps); m = len(ps); adj = np.empty(m); run = 0
    for r, i in enumerate(o):
        run = max(run, (m - r) * ps[i]); adj[i] = min(1, run)
    return adj


def main():
    d = load()
    models = [m[1] for m in MODELS if m[1] in d.model.unique()]
    sizes = {m[1]: m[2] for m in MODELS}
    print("models:", models, "rows:", len(d))
    summary = {}

    # ---------- per-condition descriptive table (per task + pooled) ----------
    rows = []
    for (m, t, c), g in d.groupby(["model", "task", "cond"]):
        rows.append(dict(model=m, task=t, cond=c, acc=g.acc.mean(), predT=g.predT.mean(),
                         pT=g.pT.mean(), ece=ece(g.conf.values, g.acc.values),
                         entropy=g.entropy.mean(), conf=g.conf.mean()))
    desc = pd.DataFrame(rows)
    desc.to_csv(f"{RES}/per_task_condition_metrics.csv", index=False)

    # ---------- Δ (target shift) with CIs + tests ----------
    itm = item_deltas(d, "predT")
    itm_soft = item_deltas(d, "pT")
    itm_acc = item_deltas(d, "acc")
    shift_rows, pvals = [], []
    boots = {}
    for m in models:
        for c in CONDS:
            sub = itm[itm.model == m]
            pt, lo, hi, bs = boot_task_mean(sub, c)
            boots[(m, c)] = bs
            # per-item mean over tasks for sign-flip: center tasks equally by using all items
            p = signflip_p(sub[c].values)
            ps, _, _, _ = boot_task_mean(itm_soft[itm_soft.model == m], c, B=1)
            pa, alo, ahi, _ = boot_task_mean(itm_acc[itm_acc.model == m], c)
            per_task = sub.groupby("task")[c].mean().to_dict()
            shift_rows.append(dict(model=m, size=sizes[m], cond=c, delta_pp=100 * pt,
                                   lo=100 * lo, hi=100 * hi, soft_delta_pp=100 * ps,
                                   dacc_pp=100 * pa, dacc_lo=100 * alo, dacc_hi=100 * ahi, p=p,
                                   **{f"delta_{t}": 100 * per_task.get(t, np.nan) for t in TASKS}))
            pvals.append(p)
    shifts = pd.DataFrame(shift_rows)
    shifts["p_holm"] = holm(shifts.p.values)
    shifts.to_csv(f"{RES}/target_shift.csv", index=False)
    print("\n=== Target shift Δ (pp) toward manipulated label, equal-task mean ===")
    print(shifts[["model", "cond", "delta_pp", "lo", "hi", "soft_delta_pp", "dacc_pp", "p_holm"]
                 + [f"delta_{t}" for t in TASKS]].round(2).to_string(index=False))

    # ranking of conditions per model
    rank = shifts.pivot(index="model", columns="cond", values="delta_pp").loc[models]
    print("\nΔ pivot (pp):\n", rank.round(1))

    # ---------- skew decomposition: gold==T vs gold!=T ----------
    k = ["model", "task", "test_id", "T"]
    w = d.pivot_table(index=k + ["gold"], columns="cond", values="acc").reset_index()
    w["is_T"] = w.gold == w["T"]
    dec = w.groupby(["model", "is_T"])[["clean", "skew", "recency", "length"]].mean().unstack().round(3)
    print("\nAccuracy split by gold==T:\n", dec)
    dec.to_csv(f"{RES}/acc_by_gold_is_T.csv")

    # ---------- noise floor: prediction flips between two independent clean draws ----------
    cl = d[(d.cond == "clean") & (d["T"] <= 1)].pivot_table(index=["model", "task", "test_id"], columns="T", values="pred")
    flips = (cl[0] != cl[1]).groupby(["model", "task"]).mean().unstack() * 100
    flips["mean"] = flips.mean(1)
    flips = flips.loc[models]
    flips.to_csv(f"{RES}/clean_resample_flip_rate.csv")
    print("\nFlip rate (%) between two independent balanced demo sets:\n", flips.round(1))

    # ---------- scale slopes (Qwen ladder) ----------
    qwen = [m for m in models if m.startswith("Qwen") and not m.endswith("base")]
    slope_rows = []
    if len(qwen) >= 3:
        x = np.log10([sizes[m] for m in qwen])
        for c in CONDS:
            y = shifts.set_index(["model", "cond"]).loc[[(m, c) for m in qwen], "delta_pp"].values
            sl = np.polyfit(x, y, 1)[0]
            bs = np.array([np.polyfit(x, 100 * np.array([boots[(m, c)][b] for m in qwen]), 1)[0]
                           for b in range(B)])
            # CI reflects item-sampling noise only (bootstrap per model), not model-to-model variation
            slope_rows.append(dict(cond=c, slope_pp_per_decade=sl, lo=np.percentile(bs, 2.5),
                                   hi=np.percentile(bs, 97.5),
                                   spearman_rho=pd.Series(y).corr(pd.Series(x), method="spearman")))
        slopes = pd.DataFrame(slope_rows)
        slopes.to_csv(f"{RES}/scale_slopes.csv", index=False)
        print("\nScale slopes (Qwen, pp per 10x params):\n", slopes.round(2))

    # ---------- calibration ----------
    cal_rows = []
    for m in models:
        for c in ["clean"] + CONDS:
            per = desc[(desc.model == m) & (desc.cond == c)]
            cal_rows.append(dict(model=m, cond=c, ece=per.ece.mean(), acc=per.acc.mean(),
                                 entropy=per.entropy.mean(), conf=per.conf.mean()))
    cal = pd.DataFrame(cal_rows)
    # bootstrap CI on ECE(cond)-ECE(clean), resampling items within task (paired).
    # Per-item sufficient statistics per confidence bin make each resample a matrix product.
    NB, BE = 10, 1000
    def item_bin_stats(g, ids):
        pos = {i: k for k, i in enumerate(ids)}
        r = g.test_id.map(pos).values
        b = np.minimum((g.conf.values * NB).astype(int), NB - 1)
        S = np.zeros((3, len(ids), NB))
        np.add.at(S[0], (r, b), 1); np.add.at(S[1], (r, b), g.conf.values); np.add.at(S[2], (r, b), g.acc.values)
        return S
    def ece_w(S, w):
        n, sc, sa = (w @ S[0]), (w @ S[1]), (w @ S[2])  # (B, NB)
        return np.abs(sa - sc).sum(1) / n.sum(1)
    dci = []
    for m in models:
        gm = d[d.model == m]
        W, stats = {}, {}
        for t in TASKS:
            gt = gm[gm.task == t]; ids = np.sort(gt.test_id.unique())
            W[t] = np.stack([np.bincount(rng.integers(0, len(ids), len(ids)), minlength=len(ids)) for _ in range(BE)]).astype(float)
            for c in ["clean"] + CONDS:
                stats[(t, c)] = item_bin_stats(gt[gt.cond == c], ids)
        for c in CONDS:
            diffs = np.mean([ece_w(stats[(t, c)], W[t]) - ece_w(stats[(t, "clean")], W[t]) for t in TASKS], 0)
            pt = cal[(cal.model == m) & (cal.cond == c)].ece.item() - cal[(cal.model == m) & (cal.cond == "clean")].ece.item()
            dci.append(dict(model=m, cond=c, dECE=pt, lo=np.percentile(diffs, 2.5), hi=np.percentile(diffs, 97.5)))
    dci = pd.DataFrame(dci)
    cal.to_csv(f"{RES}/calibration.csv", index=False)
    dci.to_csv(f"{RES}/calibration_delta.csv", index=False)
    print("\nCalibration (equal-task mean):\n", cal.pivot(index="model", columns="cond", values="ece").loc[models].round(3))
    print("\nΔECE vs clean with CI:\n", dci.round(3).to_string(index=False))
    print("\nEntropy:\n", cal.pivot(index="model", columns="cond", values="entropy").loc[models].round(3))
    print("\nAccuracy:\n", cal.pivot(index="model", columns="cond", values="acc").loc[models].round(3))

    # ---------- length interaction ----------
    w = d.pivot_table(index=["model", "task", "test_id", "T", "long_test"], columns="cond", values="predT").reset_index()
    w["dlen"] = w["length"] - w["clean"]
    li = w.groupby(["model", "long_test"]).dlen.mean().unstack() * 100
    li.columns = ["short_test", "long_test"]
    li["long_minus_short"] = li.long_test - li.short_test
    # bootstrap CI for long-minus-short
    lis = []
    for m in models:
        g = w[w.model == m]
        it = g.groupby(["task", "test_id", "long_test"]).dlen.mean().reset_index()
        L, S = it[it.long_test].dlen.values, it[~it.long_test].dlen.values
        bs = [L[rng.integers(0, len(L), len(L))].mean() - S[rng.integers(0, len(S), len(S))].mean() for _ in range(B)]
        lis.append((np.percentile(bs, 2.5) * 100, np.percentile(bs, 97.5) * 100))
    li = li.loc[models]
    li["lo"], li["hi"] = [a for a, _ in lis], [b for _, b in lis]
    li.to_csv(f"{RES}/length_interaction.csv")
    print("\nLength-condition Δ by test length (pp):\n", li.round(2))

    # ---------- figures ----------
    colors = {"skew": "#3b6fb6", "recency": "#d9822b", "primacy": "#5aa36c", "length": "#a35aa3"}
    # Fig 1: Δ by condition × model
    fig, ax = plt.subplots(figsize=(10, 4.2))
    n = len(CONDS); wdt = 0.8 / n
    for j, c in enumerate(CONDS):
        s = shifts[shifts.cond == c].set_index("model").loc[models]
        xs = np.arange(len(models)) + (j - (n - 1) / 2) * wdt
        ax.bar(xs, s.delta_pp, wdt, color=colors[c], label=c,
               yerr=[s.delta_pp - s.lo, s.hi - s.delta_pp], capsize=2, error_kw=dict(lw=0.8))
    ax.axhline(0, color="k", lw=0.6)
    ax.axhspan(15, 25, color=colors["skew"], alpha=0.08)
    ax.axhspan(8, 15, color=colors["recency"], alpha=0.08)
    ax.set_xticks(range(len(models))); ax.set_xticklabels(models, rotation=15)
    ax.set_ylabel("Δ P(pred = target) vs clean (pp)")
    ax.set_title("Shift toward the manipulated label (mean of 4 tasks, 95% bootstrap CI)\n"
                 "shaded: hypothesised ranges (skew 15–25, recency 8–15)")
    ax.legend(ncol=4, frameon=False)
    fig.tight_layout(); fig.savefig(f"{FIG}/fig1_target_shift.png", dpi=150); plt.close(fig)

    # Fig 2: scale curves per task (Qwen)
    if len(qwen) >= 2:
        fig, axes = plt.subplots(1, 5, figsize=(16, 3.4), sharey=True)
        xs = [sizes[m] for m in qwen]
        for ax, t in zip(axes, ["mean"] + TASKS):
            for c in CONDS:
                s = shifts[shifts.cond == c].set_index("model").loc[qwen]
                y = s.delta_pp if t == "mean" else s[f"delta_{t}"]
                ax.plot(xs, y, "o-", color=colors[c], label=c)
                if t == "mean":
                    ax.fill_between(xs, s.lo, s.hi, color=colors[c], alpha=0.12)
            ax.set_xscale("log"); ax.axhline(0, color="k", lw=0.5)
            ax.set_title("4-task mean" if t == "mean" else t); ax.set_xlabel("params (B)")
        axes[0].set_ylabel("Δ target rate (pp)"); axes[0].legend(frameon=False, fontsize=8)
        fig.suptitle("Bias vs. model scale (Qwen2.5-Instruct ladder)", y=1.02)
        fig.tight_layout(); fig.savefig(f"{FIG}/fig2_scale.png", dpi=150, bbox_inches="tight"); plt.close(fig)

    # Fig 3: ΔECE and accuracy
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
    for j, c in enumerate(CONDS):
        s = dci[dci.cond == c].set_index("model").loc[models]
        xs = np.arange(len(models)) + (j - (n - 1) / 2) * wdt
        axes[0].bar(xs, 100 * s.dECE, wdt, color=colors[c], label=c,
                    yerr=[100 * (s.dECE - s.lo), 100 * (s.hi - s.dECE)], capsize=2, error_kw=dict(lw=0.8))
        a = cal[cal.cond == c].set_index("model").loc[models].acc.values - cal[cal.cond == "clean"].set_index("model").loc[models].acc.values
        axes[1].bar(xs, 100 * a, wdt, color=colors[c], label=c)
    for ax, yl in zip(axes, ["ΔECE vs clean (pp)", "Δ accuracy vs clean (pp)"]):
        ax.axhline(0, color="k", lw=0.6); ax.set_xticks(range(len(models)))
        ax.set_xticklabels(models, rotation=20, fontsize=8); ax.set_ylabel(yl)
    axes[0].legend(ncol=2, frameon=False, fontsize=8)
    axes[0].set_title("Calibration error change"); axes[1].set_title("Accuracy change")
    fig.tight_layout(); fig.savefig(f"{FIG}/fig3_calibration_accuracy.png", dpi=150); plt.close(fig)
    print("figures written")


if __name__ == "__main__":
    main()
