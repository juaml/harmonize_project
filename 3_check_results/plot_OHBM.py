# %%
import numpy as np
import pandas as pd
from pathlib import Path
import sys
from sklearn.metrics import (
    mean_absolute_error,
)
import matplotlib.pyplot as plt
import seaborn as sbn

dir_path = '/home/nnieto/Nico/Harmonization/harmonize_project/3_check_results/'
__file__ = dir_path+'plot_results_cv.py'
to_append = Path(__file__).resolve().parent.parent.as_posix()
sys.path.append(to_append)

from lib.utils import extract_experiment_data

exp_dir = "/home/nnieto/Nico/Harmonization/results_r8"
experiments_to_check = {
    'test_all_regression_r8'
}

results_df = extract_experiment_data(exp_dir, experiments_to_check)
# plot_grup_barplot(results_df, True, True)

data = results_df
data["site"].replace({"1000Gehirne": "1000Brains",
                      "ID1000": "AOMIC-ID1000"}, inplace=True)


data["harmonize_mode"].replace({"pretend": "JuHarmonize",
                                "target": "Leakage",
                                "none": "None"}, inplace=True)

harm_modes = ["JuHarmonize", "Leakage", "None"]

if harm_modes is None:
    harm_modes = np.unique(data["harmonize_mode"]).tolist()


final_stat = []
for mode in harm_modes:
    resut_mode = data[data["harmonize_mode"] == mode]
    predicted_age = resut_mode["y_pred"]
    true_age = resut_mode["y_true"]
    error_data = mean_absolute_error(true_age, np.round(predicted_age))
    final_stat = np.append(final_stat, error_data)

to_sort = [harm_modes, final_stat]
df = pd.DataFrame(to_sort, index=["method", "value"])
df = df.sort_values(by="value", axis=1)
df = df.T
sort_mode = df["method"]


data["y_diff"] = data["y_true"]-data["y_pred"]

g = sbn.catplot(
    data=data, kind="boxen",
    x="site", y="y_diff", hue="harmonize_mode",
    height=10, hue_order=sort_mode, legend_out=True
)
g.set_axis_labels("", "Age prediction difference [years]")
g.legend.set_title("Harmonization Schemes")
sbn.move_legend(g, "upper right", bbox_to_anchor=(0.90, 0.98), frameon=False)

plt.grid(alpha=0.5, axis="y", c="black")
# %%
