import spacy
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline

_nlp = spacy.load("en_core_web_sm")
_sbert = SentenceTransformer("all-MiniLM-L6-v2")
_nli = pipeline("text-classification", model="facebook/bart-large-mnli", top_k=None)

def _lem(tokens):
    return {t.lemma_.lower() for t in tokens if not t.is_stop and not t.is_punct}

def similarity(student, ref):
    e = _sbert.encode([student, ref], convert_to_tensor=True)
    return float(util.cos_sim(e[0], e[1]))

def coverage(student, ref):
    s_lem = _lem(_nlp(student))
    matched, missing, seen = [], [], set()
    for ch in _nlp(ref).noun_chunks:
        lem = frozenset(_lem(ch))
        if not lem or lem in seen:
            continue
        seen.add(lem)
        (matched if lem <= s_lem else missing).append(ch.text.lower())
    total = len(matched) + len(missing)
    return (len(matched) / total if total else 0.0), matched, missing

def nli_probs(student, ref):
    r = _nli({"text": student, "text_pair": ref}, truncation=True)
    if isinstance(r[0], list):
        r = r[0]
    p = {d["label"].lower(): d["score"] for d in r}
    return p.get("entailment", 0.0), p.get("neutral", 0.0), p.get("contradiction", 0.0)

def features(student, ref):
    sim = similarity(student, ref)
    cov, matched, missing = coverage(student, ref)
    ent, neu, con = nli_probs(student, ref)
    f = {"sim": sim, "coverage": cov, "entail": ent, "contra": con}
    return f, matched, missing, (ent, neu, con)
