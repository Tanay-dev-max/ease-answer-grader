import pandas as pd, joblib
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

df = pd.read_csv("data/features.csv")
cols = ["sim", "coverage", "entail", "contra"]
Xtr, Xte, ytr, yte = train_test_split(df[cols], df["human"], test_size=0.2, random_state=42)

def report(name, pred):
    print(f"{name:30s} Pearson={pearsonr(yte, pred)[0]:.3f}  "
          f"Spearman={spearmanr(yte, pred)[0]:.3f}  MAE={mean_absolute_error(yte, pred):.3f}")

base = Ridge(alpha=1.0).fit(Xtr[["sim"]], ytr)
report("Similarity only (baseline)", base.predict(Xte[["sim"]]))

model = Ridge(alpha=1.0).fit(Xtr, ytr)
report("EASE fusion (4 signals)", model.predict(Xte))
print("Learned weights:", dict(zip(cols, model.coef_.round(3))))

joblib.dump(model, "ease_model.joblib")
print("model saved")
