import streamlit as st, joblib, numpy as np, pandas as pd
from sentence_transformers import util
from pipeline import features, _sbert

MAXS = 5
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
    X = pd.DataFrame([[f["sim"], f["coverage"], f["entail"], f["contra"]]],
                     columns=["sim", "coverage", "entail", "contra"])
    score = float(np.clip(model.predict(X)[0], 0, MAXS)) / MAXS * 10

    st.header(f"Score: {score:.1f} / 10")
    c1, c2, c3 = st.columns(3)
    c1.metric("Semantic similarity", f"{f['sim']:.3f}")
    c2.metric("Concept coverage", f"{f['coverage']*100:.0f}%")
    c3.metric("Contradiction prob.", f"{con:.2f}")
    st.write(f"**NLI:** entailment {ent:.2f} | neutral {neu:.2f} | contradiction {con:.2f}")

    st.subheader("Concepts")
    st.success("Matched: " + (", ".join(matched) or "none"))
    st.error("Missing: " + (", ".join(missing) or "none"))

    st.subheader("Word importance")
    imp = word_importance(stu, ref)
    mx = max([abs(v) for _, v in imp] + [1e-6])
    html = " ".join(
        f"<span style='background:rgba(28,114,147,{max(v,0)/mx:.2f});padding:3px 6px;"
        f"border-radius:4px'>{w}</span>" for w, v in imp)
    st.markdown(html, unsafe_allow_html=True)

    st.subheader("Feedback")
    st.write("You covered all the key concepts." if not missing
             else "Your answer is missing: " + ", ".join(missing) + ".")