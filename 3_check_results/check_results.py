from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    r2_score,
)


experiments_to_check = {
    'test_HCP_IXI_oosENKI': accuracy_score
}


for experiment, scoring in experiments_to_check.items():
    in_path = Path("/data/project/harmonize/results") / experiment

    all_dfs = []
    for t_fname in in_path.glob('*.csv'):
        all_dfs.append(pd.read_csv(t_fname, sep=';'))

    results_df = pd.concat(all_dfs)
    if 'fold' in results_df.columns:
        summary_df = results_df.groupby(
            ['harmonize_mode', 'fold']).apply(
                lambda x: scoring(x.y_true, x.y_pred)).groupby(
                    'harmonize_mode').agg(mean=np.mean, count=len)
    else:
        pass

    print(f"Experiment: {experiment}")
    print(summary_df)
    print("\n")
