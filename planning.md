# Planning: In-Context Learning Label Bias

## Motivation & Novelty Assessment

### Why This Research Matters
Few-shot prompting is the default way practitioners adapt LLMs to classification, and the demonstrations are often picked ad hoc. If superficial demonstration statistics (label counts, order, length) move predictions, a deployed classifier can quietly change behaviour depending on which examples were pasted into the prompt. Knowing how large each bias is, and whether it shrinks with scale, tells practitioners which properties of a demonstration set they actually need to control.

### Gap in Existing Work
(No literature review was pre-gathered for this workspace: papers/, datasets/ and code/ were missing. The background below comes from known work.) Zhao et al. (2021, *Calibrate Before Use*) documented majority-label, recency and common-token bias in GPT-3, mostly with small numbers of prompts on one model family, and proposed contextual calibration. Fei et al. (2023) added "domain-label bias" and domain-context calibration. Min et al. (2022) showed that label correctness matters less than format. Lu et al. (2022) showed that order sensitivity is large. Most of these studies vary one factor at a time on older GPT-3-era models. None (a) measure frequency, recency, position *and* length-label correlation side by side with the same protocol, (b) do so across a dense scale ladder (0.5B to 14B) of a single modern instruction-tuned family, or (c) report calibration and entropy for each bias type.

### Our Novel Contribution
A controlled, fully paired benchmark of four demonstration manipulations (frequency skew, recency, primacy/label-position, label-length correlation) against a clean baseline on 4 NLU tasks and 5 model sizes. The manipulation's target label is rotated through every label, so a model's intrinsic label prior cancels out and the "shift toward the target label" can be read directly. We report accuracy, calibration (ECE) and entropy.

### Experiment Justification
- Exp 1 (bias magnitude by type): estimates the shift toward the target label for each manipulation. This tests the claimed ordering skew > recency > length.
- Exp 2 (scale ladder 0.5B→14B): tests "larger models correct skew but amplify recency" and "length bias is hard to overcome with scale".
- Exp 3 (calibration/entropy): tests "ECE increases monotonically with all bias types".
- Exp 4 (length interaction): checks whether the length manipulation teaches a length→label *rule* (long test inputs pushed toward T) rather than a blanket prior shift.

## Research Question
How much does each type of spurious demonstration statistic shift an LLM's few-shot predictions toward the favoured label, and how does this depend on model scale? Types: label frequency, recency, label position (primacy), and label-length correlation.

## Hypothesis Decomposition
- H1: skew produces a 15–25 pp shift toward the majority label, the largest of all types.
- H2: recency produces an 8–15 pp shift, second largest.
- H3: length correlation produces a 5–10 pp shift that does not shrink with scale.
- H4: the skew shift decreases with scale, and the recency shift increases with scale.
- H5: ECE(cond) > ECE(clean) for every bias type.

## Proposed Methodology
### Approach
- Models: local Qwen2.5-Instruct models at 0.5B, 1.5B, 3B, 7B and 14B, run in vLLM. No API keys are available on this host and the idea specifies local models. vLLM is used instead of Ollama because it gives exact label log-probs.
- Prompts are raw completion-style text with a one-line task instruction and k=10 demonstrations.
- Prediction is the argmax of the next-token distribution restricted to the label words and renormalised over them. This avoids parsing failures and gives probabilities for calibration.
- Tasks:
  - SST-2 (sentiment, 2-way)
  - RTE (NLI, 2-way)
  - AG News (topic, 4-way)
  - MRPC (paraphrase, 2-way)
- Each task has 200 gold-balanced test items from the validation/test split. Demos are drawn from the train split.

### Conditions (k = 10, target label T rotated over all labels)
| cond | label counts | order | lengths |
|---|---|---|---|
| clean | balanced (5/5; 3/3/2/2) | random | random |
| skew | 8 T / 2 other | random | random |
| recency | balanced | last 2 = T | random |
| primacy (label-position) | balanced | first 2 = T | random |
| length | balanced | random | T demos from longest third, others from shortest third |

"Label-position correlation (label always near sentence start)" is interpreted at the *prompt* level: T demonstrations occupy the first positions (primacy). This is the counterpart of recency. The deviation is documented.

There are 10,000 prompts per model (4 tasks × 200 items × |L| targets × 5 conditions). The same prompts are used for all models, so comparisons are paired.

### Baselines
The clean balanced demonstration set, with its own fresh sample per (item, T). The no-bias reference rates are P(pred=T | clean), which equals 1/|L| on average.

### Evaluation Metrics
- Target shift Δ = P(pred=T | cond) − P(pred=T | clean), in percentage points. This is the primary bias measure.
- Accuracy and the accuracy drop relative to clean.
- ECE (10 equal-width bins, max-prob confidence).
- Mean predictive entropy (normalised by log|L|).
- Mean probability mass on T (a soft version of Δ).
- For the length condition: Δ on long test inputs (above the median) vs short ones.

### Statistical Analysis Plan
- 95% CIs come from a cluster bootstrap over test items (2,000 resamples).
- Paired sign-flip permutation tests on the per-item Δ, with Holm correction across conditions × models.
- Scale trend: OLS slope of Δ on log10(params), with a CI from bootstrapping over items.
- Calibration: the ECE difference from clean, with a bootstrap CI.

## Expected Outcomes
H1 is supported if Δ_skew lies in [15, 25] and exceeds every other Δ. H4 is supported if the skew slope is negative and the recency slope is positive (CI excluding 0). H5 is supported if ECE(cond) − ECE(clean) > 0 for all conditions in most models.

## Timeline
- Setup and prompt building: 30 min.
- Model runs: about 2–3 h, constrained by the shared GPUs.
- Analysis: 45 min.
- Report: 45 min.

## Potential Challenges
- Shared GPUs have only about 5 GB free on 3 of 4 cards. Mitigation: run 14B first on the free card, run small models on the slices, and use the AWQ fallback for 3B/7B if needed.
- Ceiling effects on SST-2/AG News for large models would compress shifts. We report per-task results.
- Lengths are confounded with content difficulty. We report the length interaction to tell a rule from a prior.

## Success Criteria
All 5 models are run on all conditions. Each hypothesis component is evaluated with CIs and an explicit supported/refuted verdict.
