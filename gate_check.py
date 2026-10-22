import pandas as pd, joblib, numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

df = pd.read_csv("data/features.csv")
cols = ["sim", "coverage", "entail", "contra"]
Xtr, Xte, ytr, yte = train_test_split(df[cols], df["human"], test_size=0.2, random_state=42)
model = joblib.load("ease_model.joblib")

hi = df["contra"] > 0.5
print("answers with contradiction prob > 0.5:", int(hi.sum()), "of", len(df))
print("mean human score | contra>0.5:", round(df.loc[hi, "human"].mean(), 2),
      "| others:", round(df.loc[~hi, "human"].mean(), 2))

base = model.predict(Xte)
for cap in [None, 3.0, 2.0]:
    p = base.copy()
    mask = Xte["contra"].values > 0.5
    if cap is not None:
        p[mask] = np.minimum(p[mask], cap)
    p = np.clip(p, 0, 5)
    print(f"cap={cap}: gated={int(mask.sum() if cap else 0)} "
          f"Pearson={pearsonr(yte, p)[0]:.3f} Spearman={spearmanr(yte, p)[0]:.3f} "
          f"MAE={mean_absolute_error(yte, p):.3f}")