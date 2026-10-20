import sys
import pandas as pd
from tqdm import tqdm
from pipeline import features

REF, STU, SCORE = "instructor_answer", "student_answer", "score_avg"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 600

df = pd.read_csv("data/mohler.csv")
df = df.dropna(subset=[REF, STU, SCORE]).sample(n=min(N, len(df)), random_state=42)

rows = []
for _, r in tqdm(df.iterrows(), total=len(df)):
    f = features(str(r[STU]), str(r[REF]))[0]
    f["human"] = r[SCORE]
    rows.append(f)

pd.DataFrame(rows).to_csv("data/features.csv", index=False)
print("saved data/features.csv with", len(rows), "rows")