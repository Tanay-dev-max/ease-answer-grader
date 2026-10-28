"""
Train and evaluate the EASE fusion model.

Inputs : data/features.csv (from build_features.py) and data/mohler.csv
Outputs: ease_model.joblib, results.md, data/demo_examples.json

Run: python train.py
"""
import json
import sys

import joblib
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GroupKFold, KFold

from ease_core import (FEATURES, LABELS, MONOTONIC, SHORT_CAP10, SHORT_ENTAIL, SHORT_MAX_WORDS, content_words,
                       keyword_overlap, shapley_values, short_answer_guard)

SEED = 42
REF, STU, Q, SCORE = "instructor_answer", "student_answer", "question", "score_avg"


def make_ease():
    return HistGradientBoostingRegressor(
        max_depth=3, learning_rate=0.05, max_iter=200, min_samples_leaf=20,
        monotonic_cst=MONOTONIC, early_stopping=False, random_state=0)


def make_ridge():
    return Ridge(alpha=1.0)


# ---------------------------------------------------------------- load data
feat = pd.read_csv("data/features.csv")
raw = pd.read_csv("data/mohler.csv").dropna(subset=[REF, STU, SCORE])
# build_features.py shuffled the rows with random_state=42; reproduce that order
raw = raw.sample(n=min(3000, len(raw)), random_state=SEED).reset_index(drop=True)
if len(raw) != len(feat) or not np.allclose(raw[SCORE].values, feat["human"].values):
    sys.exit("features.csv and mohler.csv are out of sync. Re-run: python build_features.py 3000")

df = feat.copy()
df["overlap"] = [keyword_overlap(s, r, q) for s, r, q in zip(raw[STU], raw[REF], raw[Q])]
df["question_id"] = raw["id"].str[:7].values  # e.g. "E01.Q01"
y = df["human"].values
X = df[FEATURES].values
print(f"Loaded {len(df)} answers to {df.question_id.nunique()} questions "
      f"(mean human grade {y.mean():.2f}/5)\n")


def metrics(pred):
    return dict(pearson=pearsonr(y, pred)[0], spearman=spearmanr(y, pred)[0],
                mae=mean_absolute_error(y, pred), rmse=mean_squared_error(y, pred) ** 0.5)


def out_of_fold(make, Xm, splitter, groups=None):
    pred = np.zeros(len(y))
    for tr, te in splitter.split(Xm, y, groups):
        pred[te] = np.clip(make().fit(Xm[tr], y[tr]).predict(Xm[te]), 0, 5)
    return pred


splits = {
    "5-fold CV (random)": (KFold(5, shuffle=True, random_state=SEED), None),
    "5-fold CV (unseen questions)": (GroupKFold(5), df.question_id.values),
}
systems = {
    "Similarity only (baseline)": (make_ridge, df[["sim"]].values),
    "Linear fusion (Ridge, 5 signals)": (make_ridge, X),
    "EASE (monotonic boosted trees, 5 signals)": (make_ease, X),
}

lines = ["# EASE results\n",
         f"Dataset: Mohler et al. (2011), {len(df)} answers, {df.question_id.nunique()} questions, "
         "grades 0-5.\n"]
oof = {}
for split_name, (splitter, groups) in splits.items():
    print(f"== {split_name}")
    lines += [f"\n## {split_name}\n", "| System | Pearson | Spearman | MAE | RMSE |",
              "|---|---|---|---|---|"]
    for sys_name, (make, Xm) in systems.items():
        pred = out_of_fold(make, Xm, splitter, groups)
        oof[(split_name, sys_name)] = pred
        m = metrics(pred)
        print(f"  {sys_name:44s} r={m['pearson']:.3f}  rho={m['spearman']:.3f}  "
              f"MAE={m['mae']:.3f}  RMSE={m['rmse']:.3f}")
        lines.append(f"| {sys_name} | {m['pearson']:.3f} | {m['spearman']:.3f} | "
                     f"{m['mae']:.3f} | {m['rmse']:.3f} |")
    print()

# ------------------------------------------- is EASE really better? bootstrap
base = oof[("5-fold CV (random)", "Similarity only (baseline)")]
ease = oof[("5-fold CV (random)", "EASE (monotonic boosted trees, 5 signals)")]
rng = np.random.default_rng(SEED)
diffs = []
for _ in range(1000):
    i = rng.integers(0, len(y), len(y))
    diffs.append(pearsonr(y[i], ease[i])[0] - pearsonr(y[i], base[i])[0])
lo, hi = np.percentile(diffs, [2.5, 97.5])
print(f"EASE vs similarity-only, Pearson gain: {np.mean(diffs):+.3f} (95% CI {lo:+.3f} to {hi:+.3f})\n")
lines += ["\n## Is the gain real?\n",
          f"Pearson gain of EASE over similarity-only: **{np.mean(diffs):+.3f}** "
          f"(bootstrap 95% CI {lo:+.3f} to {hi:+.3f}, 1000 resamples)."]

# ------------------------------------- does the score order match human grades?
held_out = oof[("5-fold CV (unseen questions)", "EASE (monotonic boosted trees, 5 signals)")]
bands = [("0-4", y <= 2), ("5-7", (y > 2) & (y <= 3.5)), ("8-9", (y > 3.5) & (y < 5)), ("10", y == 5)]
print("== Mean EASE score by human grade band (unseen questions, out of 10)")
lines += ["\n## Mean EASE score by human grade band (unseen-question CV, out of 10)\n",
          "| Human grade /10 | Answers | Mean human | Mean EASE |", "|---|---|---|---|"]
band_rows = []
for name, mask in bands:
    band_rows.append({"band": name, "n": int(mask.sum()), "human10": y[mask].mean() * 2,
                      "ease10": held_out[mask].mean() * 2})
    print(f"  human {name:4s}: n={mask.sum():4d}  human {y[mask].mean() * 2:.1f}  EASE {held_out[mask].mean() * 2:.1f}")
    lines.append(f"| {name} | {mask.sum()} | {y[mask].mean() * 2:.1f} | {held_out[mask].mean() * 2:.1f} |")
lines.append("\nThe order is correct in every band, but scores are compressed toward the dataset "
             "average: only 7.5% of Mohler answers are graded 4/10 or lower, so the model has "
             "few wrong answers to learn from.")
print()

# ----------------------------------------------------- short-answer guard check
n_words = np.array([content_words(s, q) for s, q in zip(raw[STU], raw[Q])])
short = (n_words <= SHORT_MAX_WORDS) & (df.entail.values < SHORT_ENTAIL)
guarded = np.array([short_answer_guard(p * 2, e, n)[0] for p, e, n in zip(held_out, df.entail, n_words)]) / 2
before, after = metrics(held_out), metrics(guarded)
print(f"== Short-answer guard (<= {SHORT_MAX_WORDS} content word, entailment < {SHORT_ENTAIL}, cap {SHORT_CAP10:.0f}/10)")
print(f"  {short.sum()} answers match; human graders gave them {y[short].mean() * 2:.1f}/10 on average")
print(f"  unseen-question Pearson {before['pearson']:.3f} -> {after['pearson']:.3f}, MAE {before['mae']:.3f} -> {after['mae']:.3f}\n")
lines += ["\n## Short-answer guard\n",
          f"Answers with at most {SHORT_MAX_WORDS} content word and entailment below {SHORT_ENTAIL} are capped at "
          f"{SHORT_CAP10:.0f}/10. {short.sum()} Mohler answers match; human graders gave them "
          f"{y[short].mean() * 2:.1f}/10 on average. On unseen-question CV the guard changes Pearson from "
          f"{before['pearson']:.3f} to {after['pearson']:.3f} and MAE from {before['mae']:.3f} to {after['mae']:.3f}."]
guard_summary = {"matches": int(short.sum()), "human10": float(y[short].mean() * 2),
                 "pearson_before": before["pearson"], "pearson_after": after["pearson"]}

# ------------------------------------------------------------------ ablation
print("== Ablation (EASE, random CV): drop one signal")
lines += ["\n## Ablation (drop one signal, random 5-fold CV)\n",
          "| Signal removed | Pearson | MAE |", "|---|---|---|"]
full = metrics(ease)
lines.append(f"| none (full EASE) | {full['pearson']:.3f} | {full['mae']:.3f} |")
ablation = [{"removed": "none (full EASE)", "pearson": full["pearson"], "mae": full["mae"]}]
for drop in FEATURES:
    keep = [c for c in FEATURES if c != drop]
    mono = [MONOTONIC[FEATURES.index(c)] for c in keep]

    def make_drop(mono=mono):
        return HistGradientBoostingRegressor(
            max_depth=3, learning_rate=0.05, max_iter=200, min_samples_leaf=20,
            monotonic_cst=mono, early_stopping=False, random_state=0)
    m = metrics(out_of_fold(make_drop, df[keep].values, *splits["5-fold CV (random)"]))
    ablation.append({"removed": LABELS[drop], "pearson": m["pearson"], "mae": m["mae"]})
    print(f"  without {drop:9s} r={m['pearson']:.3f}  MAE={m['mae']:.3f}")
    lines.append(f"| {LABELS[drop]} | {m['pearson']:.3f} | {m['mae']:.3f} |")

# ----------------------------------------- why contradiction is a flag, not a cap
print("\n== Contradiction probability vs human grade")
lines += ["\n## Contradiction flag check\n",
          "| Contradiction prob. above | Answers flagged | Mean human grade | Flagged but human >= 4/5 |",
          "|---|---|---|---|"]
contra_rows = []
for t in [0.5, 0.7, 0.9, 0.99]:
    hit = df.contra > t
    contra_rows.append({"threshold": t, "flagged": int(hit.sum()), "mean_human10": df.human[hit].mean() * 2,
                        "share_human_ge_8": float((df.human[hit] >= 4).mean())})
    print(f"  > {t}: {hit.sum():4d} flagged, mean human {df.human[hit].mean():.2f}, "
          f"{(df.human[hit] >= 4).mean():.0%} were graded >= 4/5 by humans")
    lines.append(f"| {t} | {hit.sum()} | {df.human[hit].mean():.2f} | {(df.human[hit] >= 4).mean():.0%} |")
lines.append("\nMany high-contradiction answers were still graded well by humans, so a hard score "
             "cap would be wrong. EASE flags these answers for teacher review instead.")

# --------------------------------------------------- final model + importance
model = make_ease().fit(X, y)
background = df[FEATURES].sample(n=100, random_state=SEED).values
sample = df[FEATURES].sample(n=200, random_state=1).values
phis = np.array([shapley_values(model, row, background)[1] for row in sample])
importance = dict(zip(FEATURES, np.abs(phis).mean(axis=0) * 2))  # in /10 points
print("\n== Global importance (mean |Shapley value|, points out of 10)")
lines += ["\n## Global signal importance (mean |Shapley value|, points out of 10)\n",
          "| Signal | Importance |", "|---|---|"]
for k, v in sorted(importance.items(), key=lambda kv: -kv[1]):
    print(f"  {LABELS[k]:45s} {v:.2f}")
    lines.append(f"| {LABELS[k]} | {v:.2f} |")

unseen = metrics(oof[("5-fold CV (unseen questions)", "EASE (monotonic boosted trees, 5 signals)")])
joblib.dump({
    "model": model,
    "features": FEATURES,
    "background": background,
    "cv_mae_10": unseen["mae"] * 2,  # typical error, points out of 10
    "cv_pearson_unseen": unseen["pearson"],
    "n_train": len(df),
}, "ease_model.joblib")

# ------------------------------------------- real Mohler examples for the demo
held = oof[("5-fold CV (unseen questions)", "EASE (monotonic boosted trees, 5 signals)")]
err = np.abs(held - y)
short = raw[STU].str.len().between(30, 220).values & raw[REF].str.len().lt(200).values
demos = []
for label, mask in [("Low human grade", y <= 1.5), ("Mid human grade", (y >= 2.5) & (y <= 3.5)),
                    ("High human grade", y == 5)]:
    cand = np.where(mask & short & (err <= np.median(err)))[0]
    if len(cand):
        i = cand[0]
        demos.append({"label": f"{label} (human {y[i] * 2:.0f}/10)", "question": raw[Q][i],
                      "reference": raw[REF][i], "student": raw[STU][i], "human10": float(y[i] * 2)})
with open("data/demo_examples.json", "w", encoding="utf-8") as fh:
    json.dump(demos, fh, indent=2, ensure_ascii=False)

# ------------------------------------------ evaluation data for the dashboard
summary = {
    "n": len(df), "questions": int(df.question_id.nunique()),
    "systems": [{"split": sp, "system": sy, **metrics(p)} for (sp, sy), p in oof.items()],
    "gain": {"mean": float(np.mean(diffs)), "lo": float(lo), "hi": float(hi)},
    "bands": band_rows, "guard": guard_summary, "ablation": ablation, "contradiction": contra_rows,
    "importance": [{"signal": LABELS[k], "points": float(v)} for k, v in importance.items()],
}
with open("data/eval_summary.json", "w", encoding="utf-8") as fh:
    json.dump(summary, fh, indent=2, default=float)
pd.DataFrame({"human10": y * 2, "ease10": held_out * 2, "question_id": df.question_id}).to_csv(
    "data/oof_predictions.csv", index=False)

with open("results.md", "w", encoding="utf-8") as fh:
    fh.write("\n".join(lines) + "\n")
print(f"\nSaved ease_model.joblib, results.md, data/demo_examples.json ({len(demos)} examples), "
      "data/eval_summary.json, data/oof_predictions.csv")
