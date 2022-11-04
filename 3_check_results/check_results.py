from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    r2_score,
    mean_absolute_error
)


experiments_to_check = {
    "test_all_regression": mean_absolute_error
}


for experiment, scoring in experiments_to_check.items():
    in_path = Path("/data/project/harmonize/results") / experiment

    all_dfs = []
    for t_fname in in_path.glob("*.csv"):
        df = pd.read_csv(t_fname, sep=";")
        if t_fname.name.endswith("out.csv"):
            df["kind"] = "test"
        else:
            df["kind"] = "train"
        all_dfs.append(df)

    results_df = pd.concat(all_dfs)
    if "fold" in results_df.columns:
        summary_df = results_df.groupby(
            ["kind", "harmonize_mode", "fold"]).apply(
                lambda x: scoring(x.y_true, x.y_pred)).groupby(
                    "harmonize_mode").agg(mean=np.mean, count=len)
    else:
        summary_df = results_df.groupby(
            ["kind", "harmonize_mode"]).apply(lambda x: scoring(x.y_true, x.y_pred))

    print(f"Experiment: {experiment}")
    print(summary_df)
    print("\n")
