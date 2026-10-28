"""
EASE dashboard. Run: streamlit run app.py
"""
import json
import os

import altair as alt
import joblib
import numpy as np
import pandas as pd
import streamlit as st

from ease_core import (FEATURES, LABELS, MAX_SCORE, REVIEW_CONTRA, SHORT_CAP10, content_words, feedback,
                       shapley_values, short_answer_guard, to_ten)

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


def load_json(path):
    if os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    return None


DEMOS = load_json("data/demo_examples.json") or []
SUMMARY = load_json("data/eval_summary.json")
SHORT = {k: v.split(" (")[0] for k, v in LABELS.items()}


def band_of(score):
    return "Strong" if score >= 9 else "Partial" if score >= 6.5 else "Weak"


def grade(student, reference, question="", explain=True):
    f, matched, missing, (ent, neu, con) = pipe.features(student, reference, question)
    x = np.array([f[c] for c in FEATURES])
    raw = float(model.predict(x[None, :])[0])
    model_score = to_ten(raw)
    n_words = content_words(student, question)
    score, too_short = short_answer_guard(model_score, ent, n_words)
    out = {"f": f, "x": x, "raw": raw, "model_score": model_score, "score": score,
           "band": band_of(score), "matched": matched, "missing": missing, "nli": (ent, neu, con),
           "flagged": con > REVIEW_CONTRA, "too_short": too_short, "n_words": n_words,
           "tips": feedback(f, missing, score, too_short)}
    if explain:
        out["base"], out["phi"] = shapley_values(model, x, background)
    return out


def gauge(score):
    color = "#2f8f5b" if score >= 9 else "#e8a33d" if score >= 6.5 else "#c0392b"
    return (f"<div style='font-size:3rem;font-weight:700;line-height:1'>{score:.1f}"
            f"<span style='font-size:1.4rem;opacity:.7'> / 10</span></div>"
            f"<div style='background:rgba(128,128,128,.25);border-radius:6px;height:14px;margin:10px 0'>"
            f"<div style='background:{color};width:{score * 10:.0f}%;height:14px;border-radius:6px'></div></div>")


st.title("EASE — Explainable Answer Scoring Engine")
st.caption(f"Grades descriptive answers against a reference answer and explains every grade. "
           f"Trained on {bundle['n_train']} human-graded answers (Mohler dataset).")
tab_one, tab_class, tab_perf = st.tabs(["Grade an answer", "Grade a whole class", "Model performance"])

# =====================================================================
# Tab 1: one answer
# =====================================================================
for key in ("question", "reference", "student"):
    st.session_state.setdefault(key, "")
st.session_state.setdefault("human10", None)


def load_demo():
    for d in DEMOS:
        if d["label"] == st.session_state["demo"]:
            st.session_state.update(question=d["question"], reference=d["reference"],
                                    student=d["student"], human10=d["human10"])
            return
    st.session_state["human10"] = None


def clear_human():
    st.session_state["human10"] = None


with tab_one:
    if DEMOS:
        st.selectbox("Load a real Mohler example (optional)",
                     ["Type your own"] + [d["label"] for d in DEMOS], key="demo", on_change=load_demo)
    st.text_input("Question (optional, used to ignore words copied from the question)",
                  key="question", on_change=clear_human)
    st.text_area("Reference answer", key="reference", height=90, on_change=clear_human)
    st.text_area("Student answer", key="student", height=90, on_change=clear_human)
    clicked = st.button("Grade", type="primary")
    ref, stu = st.session_state.reference.strip(), st.session_state.student.strip()
    q = st.session_state.question.strip()

    if clicked and (not ref or not stu):
        st.warning("Enter both a reference answer and a student answer.")
    elif clicked:
        with st.spinner("Grading..."):
            r = grade(stu, ref, q)
            importance = pipe.word_importance(stu, ref)
        f, (ent, neu, con) = r["f"], r["nli"]

        c_score, c_info = st.columns([1, 2])
        with c_score:
            st.markdown(gauge(r["score"]), unsafe_allow_html=True)
            st.write(f"**{r['band']} answer** · typical error ±{cv_mae:.1f}")
        with c_info:
            if st.session_state.human10 is not None:
                st.info(f"Human grader gave this answer **{st.session_state.human10:.0f}/10** "
                        f"(EASE: {r['score']:.1f}/10).")
            if r["too_short"]:
                st.warning(f"Very short answer: only {r['n_words']} content word(s) and almost no "
                           f"entailment, so the score is capped at {SHORT_CAP10:.0f}/10. On our training "
                           f"data, answers like this averaged 3.8/10 from human graders.")
            if r["flagged"]:
                st.warning(f"Flagged for teacher review: the NLI model sees a possible contradiction "
                           f"(p = {con:.2f}). The score is not changed automatically, because on our "
                           f"data contradiction alone is not a reliable sign of a wrong answer.")
            verdict = ["entailment", "neutral", "contradiction"][int(np.argmax([ent, neu, con]))]
            st.write(f"**NLI verdict: {verdict.upper()}** (entailment {ent:.2f} | neutral {neu:.2f} | "
                     f"contradiction {con:.2f})")

        cols = st.columns(5)
        for col, key, fmt in zip(cols, FEATURES, ["{:.3f}", "{:.0%}", "{:.2f}", "{:.2f}", "{:.0%}"]):
            col.metric(SHORT[key], fmt.format(f[key]))

        st.subheader("Why this score?")
        st.caption("Exact Shapley values (the method behind SHAP): how many points each signal added to "
                   "or removed from the average answer's score. The rows add up to the final score.")
        phi10 = r["phi"] * 2
        max_abs = max(np.abs(phi10).max(), 0.5)
        rows = [f"<tr><td>Average answer in the training data</td><td></td><td></td>"
                f"<td style='text-align:right'><b>{r['base'] / MAX_SCORE * 10:.2f}</b></td></tr>"]
        for key, val, p in zip(FEATURES, r["x"], phi10):
            color = "#2f8f5b" if p >= 0 else "#c0392b"
            bar = (f"<div style='background:{color};height:12px;width:{abs(p) / max_abs * 100:.0f}%;"
                   f"border-radius:3px'></div>")
            rows.append(f"<tr><td>{LABELS[key]}</td><td>{val:.2f}</td><td style='width:30%'>{bar}</td>"
                        f"<td style='text-align:right;color:{color}'><b>{p:+.2f}</b></td></tr>")
        model_out = r["raw"] / MAX_SCORE * 10
        if abs(model_out - r["model_score"]) > 0.005:
            rows.append(f"<tr><td>Model output {model_out:.2f}, limited to the 0-10 range</td><td></td>"
                        f"<td></td><td style='text-align:right'>{r['model_score'] - model_out:+.2f}</td></tr>")
        if r["too_short"]:
            rows.append(f"<tr><td>Short-answer rule (1 content word, no entailment)</td><td></td><td></td>"
                        f"<td style='text-align:right;color:#c0392b'><b>{r['score'] - r['model_score']:+.2f}</b>"
                        f"</td></tr>")
        rows.append(f"<tr><td><b>Final score</b></td><td></td><td></td>"
                    f"<td style='text-align:right'><b>{r['score']:.2f} / 10</b></td></tr>")
        st.markdown("<table style='width:100%'><tr><th>Signal</th><th>Value</th><th>Effect</th>"
                    "<th style='text-align:right'>Points</th></tr>" + "".join(rows) + "</table>",
                    unsafe_allow_html=True)

        left, right = st.columns(2)
        with left:
            st.subheader("Concepts from the reference answer")
            st.success("Matched: " + (", ".join(r["matched"]) or "none"))
            st.error("Missing: " + (", ".join(r["missing"]) or "none"))
        with right:
            st.subheader("Word importance")
            peak = max([v for _, v in importance] + [1e-6])
            st.markdown(" ".join(
                f"<span style='background:rgba(28,114,147,{max(v, 0) / peak:.2f});padding:3px 6px;"
                f"border-radius:4px;line-height:2.2'>{w}</span>" for w, v in importance),
                unsafe_allow_html=True)
            st.caption("Darker = removing the word lowers the similarity to the reference more (occlusion).")

        st.subheader("Feedback")
        for tip in r["tips"]:
            st.write("- " + tip)

# =====================================================================
# Tab 2: a whole class
# =====================================================================
with tab_class:
    st.write("Upload a CSV with columns **reference** and **student** (optional: **question**, "
             "**student_name**, **human** grade out of 10 to compare against).")
    if os.path.exists("data/sample_class.csv"):
        with open("data/sample_class.csv", "rb") as fh:
            st.download_button("Download a sample class CSV", fh.read(), "sample_class.csv", "text/csv")
    upload = st.file_uploader("Class answers (CSV)", type="csv")

    if upload is not None:
        try:
            df = pd.read_csv(upload)
        except Exception as exc:  # noqa: BLE001 - show any parsing problem to the user
            st.error(f"Could not read the CSV: {exc}")
            df = None
        if df is not None:
            df.columns = [c.strip().lower() for c in df.columns]
            missing_cols = {"reference", "student"} - set(df.columns)
            if missing_cols:
                st.error("Missing column(s): " + ", ".join(sorted(missing_cols)))
            elif st.button(f"Grade all {len(df)} answers", type="primary"):
                bar = st.progress(0.0, text="Grading...")
                results = []
                for i, row in enumerate(df.to_dict("records")):
                    stu_i, ref_i = str(row.get("student", "") or "").strip(), str(row.get("reference", "") or "").strip()
                    q_i = row.get("question", "")
                    q_i = "" if pd.isna(q_i) else str(q_i)
                    if not stu_i or stu_i.lower() == "nan" or not ref_i:
                        res = {"score": 0.0, "band": "Weak", "flag": "", "missing": "", "feedback": "No answer."}
                    else:
                        g = grade(stu_i, ref_i, q_i, explain=False)
                        res = {"score": round(g["score"], 1), "band": g["band"],
                               "flag": "Too short" if g["too_short"] else "Review" if g["flagged"] else "",
                               "missing": ", ".join(g["missing"]), "feedback": g["tips"][0]}
                    results.append(res)
                    bar.progress((i + 1) / len(df), text=f"Graded {i + 1} of {len(df)}")
                bar.empty()
                st.session_state["batch"] = pd.concat([df.reset_index(drop=True), pd.DataFrame(results)], axis=1)

    batch = st.session_state.get("batch")
    if batch is not None:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Answers graded", len(batch))
        m2.metric("Class average", f"{batch.score.mean():.1f} / 10")
        m3.metric("Flagged for review", int((batch.flag != "").sum()))
        if "human" in batch.columns:
            human = pd.to_numeric(batch.human, errors="coerce")
            if human.notna().sum() > 2 and human.nunique() > 1:
                m4.metric("Correlation with human", f"{batch.score.corr(human):.2f}")
        hist = alt.Chart(batch).mark_bar(color="#1c7293").encode(
            x=alt.X("score:Q", bin=alt.Bin(extent=[0, 10], step=1), title="EASE score (/10)"),
            y=alt.Y("count()", title="Answers")).properties(height=220)
        st.altair_chart(hist)
        show = [c for c in ["student_name", "question", "student", "score", "band", "flag", "missing",
                            "feedback", "human"] if c in batch.columns]
        st.dataframe(batch[show], hide_index=True)
        st.download_button("Download graded CSV", batch.to_csv(index=False).encode("utf-8"),
                           "graded_answers.csv", "text/csv")

# =====================================================================
# Tab 3: model performance
# =====================================================================
with tab_perf:
    if SUMMARY is None or not os.path.exists("data/oof_predictions.csv"):
        st.info("Run `python train.py` to generate the evaluation results.")
    else:
        sysdf = pd.DataFrame(SUMMARY["systems"])
        ease_unseen = sysdf[(sysdf.split.str.contains("unseen")) & (sysdf.system.str.startswith("EASE"))].iloc[0]
        st.write(f"All numbers are on held-out data from 5-fold cross-validation over "
                 f"{SUMMARY['n']} answers to {SUMMARY['questions']} questions.")
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Pearson r, unseen questions", f"{ease_unseen.pearson:.2f}")
        k2.metric("Gain over similarity alone", f"+{SUMMARY['gain']['mean']:.2f}",
                  help=f"Bootstrap 95% CI +{SUMMARY['gain']['lo']:.2f} to +{SUMMARY['gain']['hi']:.2f}")
        k3.metric("Typical error", f"±{ease_unseen.mae * 2:.1f} / 10")
        k4.metric("Spearman rank correlation", f"{ease_unseen.spearman:.2f}")

        c1, c2 = st.columns(2)
        with c1:
            st.subheader("EASE vs simpler systems")
            sysdf["System"] = sysdf.system.str.replace(r" \(.*\)", "", regex=True)
            sysdf["Test split"] = np.where(sysdf.split.str.contains("unseen"), "Unseen questions", "Random")
            sysdf["Row"] = sysdf.System + " · " + sysdf["Test split"]
            bars = alt.Chart(sysdf).mark_bar().encode(
                y=alt.Y("Row:N", sort=None, title=None, axis=alt.Axis(labelLimit=320)),
                x=alt.X("pearson:Q", title="Pearson correlation with human grades", scale=alt.Scale(domain=[0, 0.8])),
                color=alt.Color("Test split:N", scale=alt.Scale(range=["#9fc4da", "#0b3d5c"])),
                tooltip=["System", "Test split", alt.Tooltip("pearson:Q", format=".3f")]).properties(height=240)
            st.altair_chart(bars)
        with c2:
            st.subheader("Which signal matters most?")
            imp = pd.DataFrame(SUMMARY["importance"])
            st.altair_chart(alt.Chart(imp).mark_bar(color="#e8a33d").encode(
                y=alt.Y("signal:N", sort="-x", title=None, axis=alt.Axis(labelLimit=320)),
                x=alt.X("points:Q", title="Average effect on the score (points /10)"),
                tooltip=["signal", alt.Tooltip("points:Q", format=".2f")]).properties(height=240))

        st.subheader("EASE score vs human grade (held-out answers)")
        oofdf = pd.read_csv("data/oof_predictions.csv")
        oofdf["human10"] = oofdf.human10.round().astype(int)
        box = alt.Chart(oofdf).mark_boxplot(color="#1c7293", extent="min-max").encode(
            x=alt.X("human10:O", title="Human grade (/10)"),
            y=alt.Y("ease10:Q", title="EASE score (/10)", scale=alt.Scale(domain=[0, 10])))
        st.altair_chart(box.properties(height=300))
        st.caption("Scores rise with the human grade, but are compressed toward the average: only 7.5% "
                   "of training answers are graded 4/10 or lower.")

        t1, t2 = st.columns(2)
        with t1:
            st.subheader("Ablation: remove one signal")
            ab = pd.DataFrame(SUMMARY["ablation"]).rename(columns={"removed": "Signal removed",
                                                                   "pearson": "Pearson", "mae": "MAE (0-5)"})
            st.dataframe(ab.round(3), hide_index=True)
        with t2:
            st.subheader("Why contradiction is a flag, not a penalty")
            cr = pd.DataFrame(SUMMARY["contradiction"])
            cr["share_human_ge_8"] = (cr.share_human_ge_8 * 100).round().astype(int).astype(str) + "%"
            cr = cr.rename(columns={"threshold": "Contradiction above", "flagged": "Answers",
                                    "mean_human10": "Mean human /10", "share_human_ge_8": "Human gave ≥ 8/10"})
            st.dataframe(cr.round(2), hide_index=True)
