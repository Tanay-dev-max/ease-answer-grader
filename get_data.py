from datasets import load_dataset, get_dataset_config_names

NAME = "nkazi/MohlerASAG"
NEED = {"instructor_answer", "student_answer", "score_avg"}

configs = get_dataset_config_names(NAME)
print("Configs:", configs)

best = None
for cfg in configs:
    ds = load_dataset(NAME, cfg)
    for split in ds:
        df = ds[split].to_pandas()
        print(cfg, split, len(df), list(df.columns))
        if NEED <= set(df.columns) and (best is None or len(df) > len(best)):
            best = df

if best is None:
    raise SystemExit("No config had the needed columns - paste the output above to me.")

if "id" in best.columns:
    best = best.drop_duplicates(subset="id")
for c in ["instructor_answer", "student_answer"]:
    best[c] = (best[c].astype(str).str.replace("<br>", " ", regex=False)
                      .str.replace(r"\s+", " ", regex=True).str.strip())

best.to_csv("data/mohler.csv", index=False)
print("Saved", len(best), "rows to data/mohler.csv")
print(best["score_avg"].describe())