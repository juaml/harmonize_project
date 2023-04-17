# %%
from pathlib import Path
import sys

dir_path = '/home/nnieto/Nico/Harmonization/harmonize_project/3_check_results/'
__file__ = dir_path+'plot_NM.py'
to_append = Path(__file__).resolve().parent.parent.as_posix()
sys.path.append(to_append)

from lib.utils import table_generation, get_fold_acc_auc
from lib.utils import extract_experiment_data, extract_experiment_data_oos
from lib.plot_helper import plot_oos_regression, plot_oos_classification
from lib.plot_helper import plot_regression, plot_classification

# %% # %% Classification BALANCED DATA
exp_dir = "/home/nnieto/Nico/Harmonization/results_classification/"
experiments_to_check = {
    'test_classification_balanced_data_logit_stack_gssvm_pred_5repetitions'
}
# Get results
results_df = extract_experiment_data(exp_dir, experiments_to_check, True)
results_df.rename(columns={"harmonize_mode": "Harmonization Schemes"},
                  inplace=True)
data = get_fold_acc_auc(results_df)

harm_modes = ["JuHarmonize", "Leakage", "Cheat", "None"]
site_order = ["Global", "eNKI", "CamCAN", "SALD"]

plot_classification(data, harm_modes, site_order)
table = table_generation(results_df)
print(table)

# %% REGRESSION BALANCED DATA
# ###########################################################################################################################3
exp_dir = "/home/nnieto/Nico/Harmonization/results_regression/"
experiments_to_check = {'test_regression_balanced_data_rf_stack_rvr_pred'}
data = extract_experiment_data(exp_dir, experiments_to_check, train_acc=False)
data.rename(columns={"harmonize_mode": "Harmonization Schemes"},
                  inplace=True)
absolute = False
harm_modes = ["JuHarmonize", "Cheat", "Leakage", "None"]
plot_regression(data, harm_modes, absolute)
table = table_generation(data)
print(table)

# %%
exp_dir = "/home/nnieto/Nico/Harmonization/results_regression_oos/"
experiments_to_check = {'test_regression_oos_balanced_data_DLBS_full_rvr_pred_rf_stack'}
results_df = extract_experiment_data_oos(exp_dir, experiments_to_check)
results_df.rename(columns={"harmonize_mode": "Harmonization Schemes"},
                  inplace=True)
absolute = True
plot_oos_regression(results_df, harm_modes, absolute)

table = table_generation(results_df)
print(table)
# %%
exp_dir = "/home/nnieto/Nico/Harmonization/results_classification_oos/"
experiments_to_check = {'test_classification_oos_balanced_data_DLBS_full_gssvm_pred_logit_stack'}
results_df = extract_experiment_data_oos(exp_dir, experiments_to_check)
results_df.rename(columns={"harmonize_mode": "Harmonization Schemes"},
                  inplace=True)
data_final = get_fold_acc_auc(results_df)
harm_modes = ["JuHarmonize No site", "None"]
plot_oos_classification(data_final, harm_modes)
table = table_generation(results_df)
print(table)
# %%
# ###########################################################################################################################3

exp_dir = "/home/nnieto/Nico/Harmonization/results_regression_oos/"
experiments_to_check = {'test_regression_oos_balanced_data_DLBS_full_rvr_pred_rvr_stack'}
harm_modes = ["JuHarmonize No site", "None"]
absolute = True
site_list = ["Compleat DLBS", "DLBS"]
results_df = extract_experiment_data_oos(exp_dir, experiments_to_check)
results_df.rename(columns={"harmonize_mode": "Harmonization Schemes"},
                  inplace=True)
plot_oos_regression(results_df, harm_modes, absolute)
table = table_generation(results_df)
print(table)
# %%

import seaborn as sbn
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
)

exp_dir = "/home/nnieto/Nico/Harmonization/results_regression/"
experiments_to_check = {
    'test_regression_balanced_data_rvr_stack_rvr_pred'
}
# Get results
data = extract_experiment_data(exp_dir, experiments_to_check, True)
harm_modes = ["JuHarmonize", "None"]
absolute = False

data["harmonize_mode"].replace({"pretend": "JuHarmonize",
                                "none": "None"}, inplace=True)
data = data[data["harmonize_mode"].isin(harm_modes)]



# %% Ploting the difference between prediction values with between methods 
resuts_all = []
for site in np.unique(data["site"]):
    data_site = data[data["site"]==site]
    for fold in np.unique(data["fold"]):
        data_fold = data_site[data_site["fold"]==fold]

        for repeat in np.unique(data["repeat"]):
            data_repeat = data_site[data_site["repeat"]==repeat]
            Ju_harmo = data_repeat[data_repeat["harmonize_mode"] == "JuHarmonize"]
            None_data = data_repeat[data_repeat["harmonize_mode"] == "None"]
            None_data.sort_values("y_true", inplace=True)
            Ju_harmo.sort_values("y_true", inplace=True)
            None_data.reset_index(inplace=True)
            Ju_harmo.reset_index(inplace=True)
            result = Ju_harmo.copy()

            result["y_diff"] = Ju_harmo["y_pred"]-None_data["y_pred"]
            result["harmonize_mode"] = "Difference"
            true_value = Ju_harmo["y_true"]-None_data["y_true"]
            if true_value.max() != 0:
                print("ERROR")

            resuts_all.append(result)

# if absolute:
#     result["y_diff"] = np.abs(Ju_harmo["y_pred"]-None_data["y_pred"])
# else:
#     result["y_diff"] = (Ju_harmo["y_pred"]-None_data["y_pred"])

# result["y_diff_true"] = (Ju_harmo["y_true"]-None_data["y_true"])
# result["y_true_none"]= None_data["y_true"]
# result["harmonize_mode"] = "Difference JU-NONE"
resuts = pd.concat(resuts_all)
sbn.catplot(
    data=resuts, kind="violin",
    x="site", y="y_diff", hue="harmonize_mode",
    height=8, legend_out=False
)
plt.ylabel("Age prediction difference [years]")
plt.title("Harmonization Schemes")
plt.title("Age Prediction")
plt.grid(alpha=0.5, axis="y", c="black")
plt.show()

# %%
