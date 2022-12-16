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

from lib.utils import table_generation, plot_barplot, extract_experiment_data
from lib.utils import classification_table, plot_grup_barplot

# %%
exp_dir = "/home/nnieto/Nico/Harmonization/results_r8"
experiments_to_check = {
    'test_all_regression_r8'
}

results_df = extract_experiment_data(exp_dir, experiments_to_check)
# plot_grup_barplot(results_df, True, True)

data = results_df
data["harmonize_mode"].replace({"pretend": "JuHarmonize",
                                "target": "Traditional",
                                "none": "None"}, inplace=True)

harm_modes = ["JuHarmonize", "Traditional", "None"]

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
plot_barplot(results_df, True, False)
plot_barplot(results_df, True, True)

table = table_generation(results_df)
print(table)
# %%
plot_grup_barplot(results_df, False, False, ["pretend", "predict",
                                             "predict_pretend",
                                             "cheat", "target", "none"])

# %%

exp_dir = "/home/nnieto/Nico/Harmonization/results"
experiments_to_check = {
    'test_all_regression'
}

results_df = extract_experiment_data(exp_dir, experiments_to_check)
table = table_generation(results_df)
print(table)

plot_grup_barplot(results_df, True, True)
plot_grup_barplot(results_df, True, False, ["pretend", "cheat",
                                            "target", "none"])

plot_barplot(results_df, True, False)
plot_barplot(results_df, True, True)

# %%

exp_dir = "/home/nnieto/Nico/Harmonization/results_classification"
experiments_to_check = {
    'test_all_classification'
}

results_df = extract_experiment_data(exp_dir, experiments_to_check)

classification_table(results_df, ["pretend", "predict",
                                  "predict_pretend", "cheat",
                                  "target", "none"])


# %%

exp_dir = "/home/nnieto/Nico/Harmonization/results_classification"
experiments_to_check = {
    'test_all_classification_RVC'
}

results_df = extract_experiment_data(exp_dir, experiments_to_check)

classification_table(results_df)
# %%

exp_dir = "/home/nnieto/Nico/Harmonization/results_classification"
experiments_to_check = {
    'test_all_classification_RVC_TIV_remove'
}

results_df = extract_experiment_data(exp_dir, experiments_to_check)

classification_table(results_df)
# %%

exp_dir = "/home/nnieto/Nico/Harmonization/results_classification"
experiments_to_check = {
    'test_all_classification_RVC_TIV_remove_25'
}

results_df = extract_experiment_data(exp_dir, experiments_to_check)

classification_table(results_df)
# %%
exp_dir = "/home/nnieto/Nico/Harmonization/results_classification"
experiments_to_check = {
    'test_all_classification_age'
}

results_df = extract_experiment_data(exp_dir, experiments_to_check)

classification_table(results_df)
# %%

exp_dir = "/home/nnieto/Nico/Harmonization/results_classification"
experiments_to_check = {
    'test_classification_ADNI'
}

results_df = extract_experiment_data(exp_dir, experiments_to_check, False)

classification_table(results_df)
# %%
exp_dir = "/home/nnieto/Nico/Harmonization/results_classification"
experiments_to_check = {
    'test_classification_ADNI_diagnosis_svm'
}

results_df = extract_experiment_data(exp_dir, experiments_to_check, False)

classification_table(results_df)

# %%
exp_dir = "/home/nnieto/Nico/Harmonization/results_classification"
experiments_to_check = {
    'test_classification_ADNI_gender_svm'
}

results_df = extract_experiment_data(exp_dir, experiments_to_check, False)

classification_table(results_df, stats=["acc"])
# %%
exp_dir = "/home/nnieto/Nico/Harmonization/results_classification"
experiments_to_check = {
    'test_classification_ADNI_FA_svm'
}

results_df = extract_experiment_data(exp_dir, experiments_to_check, False)

classification_table(results_df, stats=["acc"])
# %%
exp_dir = "/home/nnieto/Nico/Harmonization/results_classification"
experiments_to_check = {
    'test_classification_ADNI_FA_svm'
}

results_df = extract_experiment_data(exp_dir, experiments_to_check, False)

classification_table(results_df, stats=["auc"])
# %%

exp_dir = "/home/nnieto/Nico/Harmonization/results_classification"
experiments_to_check = {
    'test_classification_ADNI_WM_svm'
}

results_df = extract_experiment_data(exp_dir, experiments_to_check, False)

classification_table(results_df, stats=["auc"])
# %%

exp_dir = "/home/nnieto/Nico/Harmonization/results_classification"
experiments_to_check = {
        'test_ADNI_WM_classification_proba'
}

results_df = extract_experiment_data(exp_dir, experiments_to_check, False)

classification_table(results_df, stats=["auc"])
# %%
exp_dir = "/home/nnieto/Nico/Harmonization/results_classification"
experiments_to_check = {
        'test_ADNI_FA_classification_proba'
}

results_df = extract_experiment_data(exp_dir, experiments_to_check, False)

classification_table(results_df, stats=["auc"])
# %%
exp_dir = "/home/nnieto/Nico/Harmonization/results_classification"
experiments_to_check = {
        'test_ADNI_FA_classification_proba_only_linear'
}

results_df = extract_experiment_data(exp_dir, experiments_to_check, False)

classification_table(results_df, stats=["auc"])
# %%

exp_dir = "/home/nnieto/Nico/Harmonization/results_classification"
experiments_to_check = {
        'test_ADNI_WM_classification_proba_only_linear'
}

results_df = extract_experiment_data(exp_dir, experiments_to_check, False)

classification_table(results_df, stats=["auc"])
# %%
exp_dir = "/home/nnieto/Nico/Harmonization/results_classification"
experiments_to_check = {
        'test_ADNI_WM_classification_new_proba'
}

results_df = extract_experiment_data(exp_dir, experiments_to_check, False)

classification_table(results_df, stats=["auc"])

# %%
