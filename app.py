"""
EASE dashboard. Run: streamlit run app.py
"""
import json
import os

import joblib
import numpy as np
import streamlit as st

from ease_core import FEATURES, LABELS, MAX_SCORE, REVIEW_CONTRA, feedback, shapley_values, to_ten

st.set_page_config(page_title="EASE", layout="wide")

if not os.path.exists("ease_model.joblib"):
    st.error("ease_model.joblib not found. Run `python train.py` first.")
    st.stop()

bundle = joblib.load("ease_model.joblib")
if "background" not in bundle:
    st.error("This ease_model.joblib is from the old train.py. Run `python train.py` again.")
    st.stop()
model, background, cv_mae = bundle["model"], bundle["background"], bundle["cv_mae_10"]


@st.cache_resource(show_spinner="Loading NLP models (first time takes about a minute)...")
def load_pipeline():
    import pipeline
    return pipeline


pipe = load_pipeline()

DEMOS = []
if os.path.exists("data/demo_examples.json"):
    with open("data/demo_examples.json", encoding="utf-8") as fh:
        DEMOS = json.load(fh)

for key in ("question", "reference", "student"):
    st.session_state.setdefault(key, "")
st.session_state.setdefault("human10", None)


def load_demo():
    choice = st.session_state["demo"]
    for d in DEMOS:
        if d["label"] == choice:
            st.session_state.update(question=d["question"], reference=d["reference"],
                                    student=d["student"], human10=d["human10"])
            return
    st.session_state["human10"] = None


def clear_human():
    st.session_state["human10"] = None


# ------------------------------------------------------------------- inputs
st.title("EASE — Explainable Answer Scoring Engine")
st.caption(f"Grades a descriptive answer against a reference answer and explains the grade. "
           f"Trained on {bundle['n_train']} human-graded answers (Mohler dataset).")

if DEMOS:
    st.selectbox("Load a real Mohler example (optional)", ["Type your own"] + [d["label"] for d in DEMOS],
                 key="demo", on_change=load_demo)
st.text_input("Question (optional, used to ignore words copied from the question)",
              key="question", on_change=clear_human)
st.text_area("Reference answer", key="reference", height=90, on_change=clear_human)
st.text_area("Student answer", key="student", height=90, on_change=clear_human)

ref, stu, q = st.session_state.reference.strip(), st.session_state.student.strip(), st.session_state.question.strip()
if not st.button("Grade", type="primary"):
    st.stop()
if not ref or not stu:
    st.warning("Enter both a reference answer and a student answer.")
    st.stop()

# ------------------------------------------------------------------ scoring
with st.spinner("Grading..."):
    f, matched, missing, (ent, neu, con) = pipe.features(stu, ref, q)
    x = np.array([f[c] for c in FEATURES])
    raw = float(model.predict(x[None, :])[0])
    score = to_ten(raw)
    base, phi = shapley_values(model, x, background)
    importance = pipe.word_importance(stu, ref)

band = "Strong" if score >= 8 else "Partial" if score >= 5 else "Weak"
c_score, c_info = st.columns([1, 2])
c_score.metric("Score", f"{score:.1f} / 10", help="Predicted human grade, rescaled from 0-5 to 0-10.")
c_score.write(f"**{band} answer** · typical error ±{cv_mae:.1f}")
if st.session_state.human10 is not None:
    c_info.info(f"Human grader gave this answer **{st.session_state.human10:.0f}/10** "
                f"(EASE: {score:.1f}/10).")
if con > REVIEW_CONTRA:
    c_info.warning(f"Flagged for teacher review: the NLI model sees a possible contradiction "
                   f"(p = {con:.2f}). The score is not changed automatically, because on our data "
                   f"contradiction alone is not a reliable sign of a wrong answer.")

cols = st.columns(5)
for col, key, fmt in zip(cols, FEATURES, ["{:.3f}", "{:.0%}", "{:.2f}", "{:.2f}", "{:.0%}"]):
    col.metric(LABELS[key].split(" (")[0], fmt.format(f[key]))
verdict = ["entailment", "neutral", "contradiction"][int(np.argmax([ent, neu, con]))]
st.write(f"**NLI verdict: {verdict.upper()}** (entailment {ent:.2f} | neutral {neu:.2f} | "
         f"contradiction {con:.2f})")

# -------------------------------------------------------- why this score?
st.subheader("Why this score?")
st.caption("Exact Shapley values (the method behind SHAP): how many points each signal added to or "
           "removed from the average answer's score. The rows add up to the final score.")
max_abs = max(np.abs(phi * 2).max(), 0.5)
rows = [f"<tr><td>Average answer in the training data</td><td></td><td></td>"
        f"<td style='text-align:right'><b>{base / MAX_SCORE * 10:.2f}</b></td></tr>"]
for key, val, p in zip(FEATURES, x, phi * 2):
    color = "#2f8f5b" if p >= 0 else "#c0392b"
    width = abs(p) / max_abs * 100
    bar = (f"<div style='background:{color};height:12px;width:{width:.0f}%;border-radius:3px'></div>")
    rows.append(f"<tr><td>{LABELS[key]}</td><td>{val:.2f}</td><td style='width:30%'>{bar}</td>"
                f"<td style='text-align:right;color:{color}'><b>{p:+.2f}</b></td></tr>")
model_out = raw / MAX_SCORE * 10
if abs(model_out - score) > 0.005:
    rows.append(f"<tr><td>Model output {model_out:.2f}, limited to the 0-10 range</td><td></td><td></td>"
                f"<td style='text-align:right'>{score - model_out:+.2f}</td></tr>")
rows.append(f"<tr><td><b>Final score</b></td><td></td><td></td>"
            f"<td style='text-align:right'><b>{score:.2f} / 10</b></td></tr>")
st.markdown("<table style='width:100%'><tr><th>Signal</th><th>Value</th><th>Effect</th>"
            "<th style='text-align:right'>Points</th></tr>" + "".join(rows) + "</table>",
            unsafe_allow_html=True)

# ---------------------------------------------------- concepts and words
left, right = st.columns(2)
with left:
    st.subheader("Concepts from the reference answer")
    st.success("Matched: " + (", ".join(matched) or "none"))
    st.error("Missing: " + (", ".join(missing) or "none"))
with right:
    st.subheader("Word importance")
    peak = max([v for _, v in importance] + [1e-6])
    st.markdown(" ".join(
        f"<span style='background:rgba(28,114,147,{max(v, 0) / peak:.2f});padding:3px 6px;"
        f"border-radius:4px;line-height:2.2'>{w}</span>" for w, v in importance),
        unsafe_allow_html=True)
    st.caption("Darker = removing the word lowers the similarity to the reference more (occlusion).")

st.subheader("Feedback")
for tip in feedback(f, missing, score):
    st.write("- " + tip)

with st.expander("How reliable is this score?"):
    st.write(f"On answers to questions it never saw during training, EASE's scores correlate with "
             f"human grades at Pearson r = {bundle['cv_pearson_unseen']:.2f}, with a typical error "
             f"of ±{cv_mae:.1f} points out of 10 (5-fold cross-validation). Scores are pulled toward "
             f"the dataset average, because few answers in the training data are wrong. Treat EASE "
             f"as an assistant to the teacher, not a replacement.")
