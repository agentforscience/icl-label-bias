"""Build controlled few-shot prompts for the ICL label-bias study.

For each task we draw a fixed evaluation set (balanced over gold labels) and,
for every (test example, target label T, condition), a fresh k=10 demonstration
set from the training split.  The "target label" T is the label favoured by the
manipulation; T is rotated through every label so that a model's prior label
preference cancels out when we average shifts.

Conditions
  clean    : balanced label counts (5/5, or 3/3/2/2 for 4-way), random order
  skew     : 8 of 10 demos carry T (80/20)
  recency  : balanced counts, the last 2 demos carry T
  primacy  : balanced counts, the first 2 demos carry T   (label-position)
  length   : balanced counts, T demos drawn from the longest third of the
             T pool, non-T demos from the shortest third of their pools
All prompts are written to datasets/prompts/prompts.jsonl and are shared by
every model, so comparisons across models are fully paired.
"""
import json, hashlib, random, os
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HUB = "/data/transformers/hub"
K = 10
N_TEST_PER_TASK = 200
CONDITIONS = ["clean", "skew", "recency", "primacy", "length"]
# supplementary dose-response conditions (separate prompt file)
SUPP_CONDITIONS = ["skew60", "skew100", "recency4"]


def _snap(repo):
    d = f"{HUB}/datasets--{repo}/snapshots"
    return os.path.join(d, sorted(os.listdir(d))[0], "data")


def load_tasks():
    """Return dict task -> (train_df, test_df, spec). Each df has text fields,
    `label` (int) and `nwords` (length used for the length manipulation)."""
    tasks = {}
    sst = _snap("stanfordnlp--sst2")
    tr = pd.read_parquet(f"{sst}/train-00000-of-00001.parquet").rename(columns={"sentence": "x"})
    te = pd.read_parquet(f"{sst}/validation-00000-of-00001.parquet").rename(columns={"sentence": "x"})
    tr["x"] = tr.x.str.strip(); te["x"] = te.x.str.strip()
    tasks["sst2"] = (tr, te, dict(
        instr="Classify the sentiment of each movie review as positive or negative.",
        fields=[("Review", "x")], answer="Sentiment", labels=["negative", "positive"]))

    ag = _snap("fancyzhx--ag_news")
    tr = pd.read_parquet(f"{ag}/train-00000-of-00001.parquet").rename(columns={"text": "x"})
    te = pd.read_parquet(f"{ag}/test-00000-of-00001.parquet").rename(columns={"text": "x"})
    tasks["agnews"] = (tr, te, dict(
        instr="Classify the topic of each news article as World, Sports, Business, or Technology.",
        fields=[("Article", "x")], answer="Topic",
        labels=["World", "Sports", "Business", "Technology"]))

    for t, instr, f1, f2, ans, labels in [
        ("rte", "Decide whether the premise entails the hypothesis (yes or no).",
         "Premise", "Hypothesis", "Entailment", ["yes", "no"]),  # 0=entailment
        ("mrpc", "Decide whether the two sentences are paraphrases of each other (yes or no).",
         "Sentence 1", "Sentence 2", "Paraphrase", ["no", "yes"]),  # 1=equivalent
    ]:
        tr = pd.read_parquet(f"{ROOT}/datasets/raw/{t}/train.parquet")
        te = pd.read_parquet(f"{ROOT}/datasets/raw/{t}/validation.parquet")
        tasks[t] = (tr, te, dict(instr=instr, fields=[(f1, "sentence1"), (f2, "sentence2")],
                                 answer=ans, labels=labels))

    for t, (tr, te, spec) in tasks.items():
        for df in (tr, te):
            df["nwords"] = sum(df[c].str.split().str.len() for _, c in spec["fields"])
        # keep demo pool to reasonable lengths so prompts stay < ~2k tokens
        lo = 5 if t == "sst2" else 1
        hi = {"sst2": 60, "agnews": 80, "rte": 80, "mrpc": 70}[t]
        tr = tr[(tr.nwords >= lo) & (tr.nwords <= hi)].reset_index(drop=True)
        te = te[(te.nwords <= hi * 1.5)].reset_index(drop=True)
        tasks[t] = (tr, te, spec)
    return tasks


def seed_of(*parts):
    return int(hashlib.md5("|".join(map(str, parts)).encode()).hexdigest()[:8], 16)


def fmt_example(row, spec, with_label=True):
    s = "".join(f"{name}: {row[col]}\n" for name, col in spec["fields"])
    s += f"{spec['answer']}:"
    if with_label:
        s += f" {spec['labels'][int(row['label'])]}\n\n"
    return s


def label_counts(n_labels, T, cond, rng):
    """Number of demos per label for a given condition."""
    if cond in ("skew", "skew60", "skew100"):
        nT = {"skew": 8, "skew60": 6, "skew100": 10}[cond]
        counts = [0] * n_labels
        counts[T] = nT
        others = [l for l in range(n_labels) if l != T]
        for l in rng.choice(others, size=K - nT, replace=len(others) < K - nT):
            counts[l] += 1
        return counts
    if n_labels == 2:
        return [5, 5]
    base = [3, 3, 2, 2]  # 4-way: random which two labels get 3
    rng.shuffle(base)
    return base


def build_demos(train, spec, T, cond, rng, length_pools):
    n_labels = len(spec["labels"])
    counts = label_counts(n_labels, T, cond, rng)
    nfix = 4 if cond == "recency4" else 2
    if n_labels == 4 and cond == "recency4":
        counts = [2, 2, 2, 2]; counts[T] = 4  # 4 T demos all at the end
    if n_labels == 4 and cond in ("recency", "primacy") and counts[T] < 2:
        # make sure T can fill the 2 designated slots (swap with a 3-label)
        j = int(np.argmax(counts)); counts[T], counts[j] = counts[j], counts[T]
    demos = []
    for l, c in enumerate(counts):
        if c == 0:
            continue
        if cond == "length":
            pool = length_pools[l]["long" if l == T else "short"]
        else:
            pool = length_pools[l]["all"]
        idx = rng.choice(pool, size=c, replace=False)
        demos += [train.iloc[i] for i in idx]
    order = rng.permutation(len(demos))
    demos = [demos[i] for i in order]
    if cond in ("recency", "primacy", "recency4"):
        tdem = [d for d in demos if int(d.label) == T]
        rest = [d for d in demos if int(d.label) != T]
        fixed, rest = tdem[:nfix], tdem[nfix:] + rest
        rest = [rest[i] for i in rng.permutation(len(rest))]
        demos = fixed + rest if cond == "primacy" else rest + fixed
    assert len(demos) == K
    return demos


def main():
    import sys
    supp = len(sys.argv) > 1 and sys.argv[1] == "supp"
    conds = SUPP_CONDITIONS if supp else CONDITIONS
    tasks = load_tasks()
    out = []
    for t, (train, test, spec) in tasks.items():
        n_labels = len(spec["labels"])
        rng = np.random.default_rng(seed_of("test", t))
        per = N_TEST_PER_TASK // n_labels
        test_idx = np.concatenate([rng.choice(np.where(test.label.values == l)[0], per, replace=False)
                                   for l in range(n_labels)])
        pools = {}
        for l in range(n_labels):
            ids = np.where(train.label.values == l)[0]
            nw = train.nwords.values[ids]
            q1, q2 = np.quantile(nw, [1 / 3, 2 / 3])
            pools[l] = dict(all=ids, short=ids[nw <= q1], long=ids[nw >= q2])
        print(t, "train pool", len(train), "test", len(test_idx),
              {l: (len(p["short"]), len(p["long"])) for l, p in pools.items()})
        for ti in test_idx:
            row = test.iloc[ti]
            for T in range(n_labels):
                for cond in conds:
                    r = np.random.default_rng(seed_of(t, ti, T, cond))
                    demos = build_demos(train, spec, T, cond, r, pools)
                    prompt = spec["instr"] + "\n\n" + "".join(fmt_example(d, spec) for d in demos) \
                        + fmt_example(row, spec, with_label=False)
                    out.append(dict(
                        task=t, test_id=int(ti), gold=int(row.label), T=T, cond=cond,
                        test_nwords=int(row.nwords), labels=spec["labels"],
                        demo_labels=[int(d.label) for d in demos],
                        demo_nwords=[int(d.nwords) for d in demos], prompt=prompt))
    path = f"{ROOT}/datasets/prompts/prompts{'_supp' if supp else ''}.jsonl"
    with open(path, "w") as f:
        for o in out:
            f.write(json.dumps(o) + "\n")
    print("wrote", len(out), "prompts ->", path)
    df = pd.DataFrame(out)
    print(df.groupby(["task", "cond"]).size().unstack())
    df["frac_T"] = [np.mean(np.array(d) == T) for d, T in zip(df.demo_labels, df["T"])]
    df["last2_T"] = [np.mean(np.array(d[-2:]) == T) for d, T in zip(df.demo_labels, df["T"])]
    df["first2_T"] = [np.mean(np.array(d[:2]) == T) for d, T in zip(df.demo_labels, df["T"])]
    df["lenT_minus_other"] = [np.mean([w for w, l in zip(nw, d) if l == T]) - np.mean([w for w, l in zip(nw, d) if l != T])
                              for nw, d, T in zip(df.demo_nwords, df.demo_labels, df["T"])]
    df["plen"] = df.prompt.str.split().str.len()
    print(df.groupby(["task", "cond"])[["frac_T", "first2_T", "last2_T", "lenT_minus_other", "plen"]].mean().round(2))


if __name__ == "__main__":
    main()
