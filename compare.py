import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

df = pd.read_csv("data/features.csv")
cols = ["sim", "coverage", "entail", "contra"]
Xtr, Xte, ytr, yte = train_test_split(df[cols], df["human"], test_size=0.2, random_state=42)

wrong = pd.DataFrame([[0.477, 0.0, 0.22, 0.36]], columns=cols)  # your wrong-answer test

print(df["human"].describe())
print("share of answers scoring below 3:", round((df["human"] < 3).mean(), 3))

models = [("Ridge", Ridge(alpha=1.0)),
          ("GradientBoosting", GradientBoostingRegressor(
              n_estimators=150, max_depth=2, learning_rate=0.05, random_state=42))]
for name, m in models:
    m.fit(Xtr, ytr)
    p = m.predict(Xte)
    print(f"{name:18s} Pearson={pearsonr(yte, p)[0]:.3f} "
          f"Spearman={spearmanr(yte, p)[0]:.3f} MAE={mean_absolute_error(yte, p):.3f} "
          f"wrong-answer score={m.predict(wrong)[0] / 5 * 10:.1f}/10")