from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    r2_score,
    mean_absolute_error
)

results_path = "/data/project/harmonize/results_kbest"

experiments_to_check = "overfit_all_regression"

scoring = mean_absolute_error
select_ks = list(range(100, 3401, 100))



all_summaries = []
for k in select_ks:
    experiment = f"{experiments_to_check}_{k}"
    in_path = Path(results_path) / experiment

    all_dfs = []
    for t_fname in in_path.glob("*.csv"):
        df = pd.read_csv(t_fname, sep=";")
        if t_fname.name.endswith("out.csv"):
            df["kind"] = "test"
        else:
            df["kind"] = "train"
        all_dfs.append(df)
    if len(all_dfs) == 0:
        continue
    results_df = pd.concat(all_dfs)
    if "fold" in results_df.columns:
        summary_df = results_df.groupby(
            ["kind", "harmonize_mode", "fold"]).apply(
                lambda x: scoring(x.y_true, x.y_pred)).groupby(
                    ["kind", "harmonize_mode"]).agg(mean=np.mean, count=len)
    else:
        summary_df = results_df.groupby(
            ["kind", "harmonize_mode"]).apply(lambda x: scoring(x.y_true, x.y_pred))

    # print(f"Experiment: {experiment}")
    # print(summary_df)
    # print("\n")
    summary_df["k"] = k
    all_summaries.append(summary_df)

all_summaries_df = pd.concat(all_summaries)
print(all_summaries_df)