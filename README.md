# EASE: Explainable Answer Scoring Engine

Grades descriptive answers by fusing three NLP signals (semantic similarity,
concept coverage, NLI entailment/contradiction) with a Ridge regression model
trained on human-graded answers, with matched/missing concepts and
word-importance shown for every grade.

## Setup
pip install -r requirements.txt
python -m spacy download en_core_web_sm

## Run
python get_data.py          # download Mohler dataset
python build_features.py 3000   # extract features
python train.py             # train fusion model, prints results
streamlit run app.py        # dashboard

## Results
(add your final Pearson / Spearman / MAE table here tomorrow)

## Dataset
Mohler et al., 2011, via Hugging Face: nkazi/MohlerASAG