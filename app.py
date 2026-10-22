import streamlit as st, joblib, numpy as np, pandas as pd
from sentence_transformers import util
from pipeline import features, _sbert

st.set_page_config(page_title="EASE", layout="wide")

MAXS = 5
GATE_P, GATE_CAP = 0.5, 2.0   # contradiction rule: cap = 2.0/5 = 4/10
COLS = ["sim", "coverage", "entail", "contra"]
LABELS = {"sim": "Semantic similarity", "coverage": "Concept coverage",
          "entail": "NLI entailment prob.", "contra": "NLI contradiction prob."}
model = joblib.load("ease_model.joblib")

def word_importance(stu, ref):
    words = stu.split()
    variants = [" ".join(words[:i] + words[i+1:]) for i in range(len(words))]
    emb = _sbert.encode([stu, ref] + variants, convert_to_tensor=True)
    base = util.cos_sim(emb[0], emb[1]).item()
    return [(w, base - util.cos_sim(emb[2+i], emb[1]).item()) for i, w in enumerate(words)]

st.title("EASE — Explainable Answer Grader")
ref = st.text_area("Reference answer", height=100)
stu = st.text_area("Student answer", height=100)

if st.button("Grade") and ref.strip() and stu.strip():
    f, matched, missing, (ent, neu, con) = features(stu, ref)
    x = [f[c] for c in COLS]
    raw_unclipped = float(model.predict(pd.DataFrame([x], columns=COLS))[0])
    raw = float(np.clip(raw_unclipped, 0, MAXS))
    gated = con > GATE_P
    final = min(raw, GATE_CAP) if gated else raw
    score = final / MAXS * 10
    model_score = raw / MAXS * 10

    # ---- score + reason ----
    st.header(f"Score: {score:.1f} / 10")
    if gated:
        st.warning(f"Contradiction detected (p={con:.2f}). The student's answer conflicts with "
                   f"the reference answer, so the score is capped at {GATE_CAP/MAXS*10:.0f}/10 "
                   f"(the model alone would have given {model_score:.1f}/10).")

    # ---- the four signals in one row ----
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Semantic similarity", f"{f['sim']:.3f}")
    c2.metric("Concept coverage", f"{f['coverage']*100:.0f}%")
    c3.metric("Entailment prob.", f"{ent:.2f}")
    c4.metric("Contradiction prob.", f"{con:.2f}")
    verdict = ["entailment", "neutral", "contradiction"][int(np.argmax([ent, neu, con]))]
    st.write(f"**NLI verdict: {verdict.upper()}** "
             f"(entailment {ent:.2f} | neutral {neu:.2f} | contradiction {con:.2f})")

    # ---- why this score ----
    st.subheader("Why this score?")
    rows = []
    for name, val, w in zip(COLS, x, model.coef_):
        rows.append({"Signal": LABELS[name], "Value": f"{val:.3f}",
                     "Learned weight": f"{w:+.3f}",
                     "Contribution (/10)": f"{w * val / MAXS * 10:+.2f}"})
    rows.append({"Signal": "Baseline (intercept)", "Value": "-", "Learned weight": "-",
                 "Contribution (/10)": f"{model.intercept_ / MAXS * 10:+.2f}"})
    rows.append({"Signal": "Model score (sum, clipped to 0-10)", "Value": "-",
                 "Learned weight": "-", "Contribution (/10)": f"{model_score:.2f}"})
    if gated:
        rows.append({"Signal": f"Contradiction rule (p > {GATE_P})", "Value": f"{con:.2f}",
                     "Learned weight": "rule, not learned",
                     "Contribution (/10)": f"cap at {GATE_CAP/MAXS*10:.1f}"})
    rows.append({"Signal": "FINAL SCORE", "Value": "-", "Learned weight": "-",
                 "Contribution (/10)": f"{score:.2f}"})
    st.table(pd.DataFrame(rows))

    # ---- concepts ----
    st.subheader("Concepts")
    st.success("Matched: " + (", ".join(matched) or "none"))
    st.error("Missing: " + (", ".join(missing) or "none"))

    # ---- word importance ----
    st.subheader("Word importance")
    imp = word_importance(stu, ref)
    mx = max([abs(v) for _, v in imp] + [1e-6])
    html = " ".join(
        f"<span style='background:rgba(28,114,147,{max(v,0)/mx:.2f});padding:3px 6px;"
        f"border-radius:4px'>{w}</span>" for w, v in imp)
    st.markdown(html, unsafe_allow_html=True)
    st.caption("Darker = the word contributes more to the similarity with the reference answer.")

    # ---- feedback ----
    st.subheader("Feedback")
    tips = []
    if gated:
        tips.append("Your answer contradicts the reference answer on a key point. Re-check the core definition.")
    if missing:
        tips.append("Missing concepts: " + ", ".join(missing) + ".")
    if not tips:
        tips.append("You covered all key concepts and your answer agrees with the reference.")
    for t in tips:
        st.write("- " + t)