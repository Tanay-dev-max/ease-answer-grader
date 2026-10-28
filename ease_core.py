"""
EASE core logic: everything that does NOT need the heavy NLP models.
Used by train.py (training/evaluation) and app.py (scoring + explanations).
"""
import re
from itertools import combinations
from math import factorial

import numpy as np

MAX_SCORE = 5.0  # Mohler grades are on a 0-5 scale; the app shows /10

# The five signals the fusion model uses, in this exact order.
FEATURES = ["sim", "coverage", "entail", "contra", "overlap"]
LABELS = {
    "sim": "Semantic similarity (Sentence-BERT)",
    "coverage": "Concept coverage (spaCy noun phrases)",
    "entail": "NLI entailment probability",
    "contra": "NLI contradiction probability",
    "overlap": "Keyword overlap (question words removed)",
}
# Monotonic constraints: +1 = a higher value can never lower the score,
# -1 = a higher value can never raise the score.
MONOTONIC = [1, 1, 1, -1, 1]

# Answers above this contradiction probability are flagged for teacher review.
# The score is NOT capped: on Mohler, a hard cap would wrongly punish many
# answers that human graders scored highly (see train.py output).
REVIEW_CONTRA = 0.9

# Short-answer guard: an answer with at most 1 content word (after removing
# question words) and almost no entailment is capped at 5/10. On Mohler, such
# answers averaged 3.8/10 from human graders; the cap leaves cross-validated
# accuracy unchanged (see train.py output).
SHORT_MAX_WORDS = 1
SHORT_ENTAIL = 0.05
SHORT_CAP10 = 5.0

_STOP = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being", "am",
    "of", "in", "on", "at", "to", "for", "from", "by", "with", "as", "into",
    "and", "or", "but", "if", "then", "so", "than", "that", "this", "these",
    "those", "it", "its", "they", "them", "their", "there", "here", "which",
    "who", "whom", "what", "when", "where", "why", "how", "can", "could",
    "will", "would", "should", "may", "might", "must", "do", "does", "did",
    "has", "have", "had", "not", "no", "yes", "you", "your", "we", "our", "i",
    "me", "my", "he", "she", "his", "her", "also", "such", "any", "all",
    "each", "some", "more", "most", "other", "only", "very", "just", "about",
}


def _tokens(text):
    out = []
    for w in re.findall(r"[a-z0-9]+", str(text).lower()):
        if w in _STOP:
            continue
        if len(w) > 4 and w.endswith("s") and not w.endswith("ss"):
            w = w[:-1]  # crude plural normalisation: "pointers" -> "pointer"
        out.append(w)
    return set(out)


def keyword_overlap(student, reference, question=""):
    """Share of the reference answer's content words that the student used.

    Question demoting: words that already appear in the question are removed
    first, so a student can't score by repeating the question back.
    """
    s, r, q = _tokens(student), _tokens(reference), _tokens(question)
    r_demoted = r - q
    if r_demoted:
        return len((s - q) & r_demoted) / len(r_demoted)
    return len(s & r) / len(r) if r else 0.0


def content_words(student, question=""):
    """Number of distinct content words in the answer that are not copied from the question."""
    return len(_tokens(student) - _tokens(question))


def short_answer_guard(score10, entail, n_words):
    """Returns (score, applied). Caps very short, non-entailing answers."""
    if n_words <= SHORT_MAX_WORDS and entail < SHORT_ENTAIL:
        return min(score10, SHORT_CAP10), score10 > SHORT_CAP10
    return score10, False


def to_ten(raw_score):
    """Model output on the 0-5 Mohler scale -> displayed score out of 10."""
    return float(np.clip(raw_score, 0.0, MAX_SCORE)) / MAX_SCORE * 10.0


def shapley_values(model, x, background):
    """Exact Shapley values (the method behind SHAP) for one prediction.

    With 5 signals there are only 2^5 = 32 feature coalitions, so we compute
    the exact values instead of approximating. Missing features are filled
    from a background sample of the training data (interventional SHAP).

    Returns (base_value, phi) where base_value + phi.sum() == model output
    for x (before clipping), both on the 0-5 scale.
    """
    x = np.asarray(x, dtype=float).reshape(-1)
    bg = np.asarray(background, dtype=float)
    n, b = len(x), len(bg)
    subsets = [s for k in range(n + 1) for s in combinations(range(n), k)]

    batch = np.repeat(bg[None, :, :], len(subsets), axis=0)  # (subsets, b, n)
    for j, s in enumerate(subsets):
        if s:
            batch[j][:, list(s)] = x[list(s)]
    preds = model.predict(batch.reshape(-1, n)).reshape(len(subsets), b).mean(axis=1)
    value = {s: v for s, v in zip(subsets, preds)}

    phi = np.zeros(n)
    for i in range(n):
        for s in subsets:
            if i in s:
                continue
            with_i = tuple(sorted(s + (i,)))
            weight = factorial(len(s)) * factorial(n - len(s) - 1) / factorial(n)
            phi[i] += weight * (value[with_i] - value[s])
    return value[()], phi


def feedback(f, missing, score10, too_short=False):
    """Short, rule-based feedback built from the signals."""
    tips = []
    if too_short:
        tips.append("Your answer is too short to show understanding. Explain the idea in a full sentence.")
    if f["contra"] > REVIEW_CONTRA:
        tips.append("Part of your answer may contradict the reference answer. "
                    "Re-check the core definition.")
    if f["sim"] < 0.3 and f["entail"] < 0.2:
        tips.append("Your answer seems to be about something different from the "
                    "question. Focus on what was asked.")
    if missing:
        tips.append("Concepts missing from your answer: " + ", ".join(missing) + ".")
    if not tips:
        if score10 >= 8:
            tips.append("Good answer: it matches the reference and covers the key concepts.")
        else:
            tips.append("Your answer is on the right track. Add more detail on the key "
                        "points of the reference answer.")
    return tips
