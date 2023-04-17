# %%
import numpy as np
import pandas as pd
from pathlib import Path
import sys
from statannot.statannot import add_stat_annotation

import matplotlib.pyplot as plt
import seaborn as sbn
from sklearn.metrics import (
    mean_absolute_error,
)
dir_path = '/home/nnieto/Nico/Harmonization/harmonize_project/3_check_results/'
__file__ = dir_path+'plot_NM.py'
to_append = Path(__file__).resolve().parent.parent.as_posix()
sys.path.append(to_append)

from lib.utils import extract_experiment_data, get_fold_acc_auc
from lib.utils import table_generation
from lib.utils import table_generation, plot_grup_barplot, extract_experiment_data, plot_barplot_classification, extract_experiment_data_oos

# %%

exp_dir = "/home/nnieto/Nico/Harmonization/results_classification/"
experiments_to_check = {
    'test_classification_Kersten_data_logit_stack_gssvm_pred_5repetitions'
}
# Get results
results_df = extract_experiment_data(exp_dir, experiments_to_check, False)

data_final = get_fold_acc_auc(results_df)
# Change to appropiated names
data_final["Harmonization Schemes"].replace({"pretend": "JuHarmonize",
                                             "target": "Leakage",
                                             "none": "None",
                                             "cheat": "Cheat"},
                                            inplace=True)

# Select methods to plot
harm_modes = ["JuHarmonize", "Leakage", "Cheat", "None"]
data_final = data_final[data_final["Harmonization Schemes"].isin(harm_modes)]
site_order = ["Global", "SITE_1", "SITE_2"]

# Plot
fig, ax = plt.subplots(1, 1, figsize=[20, 10])
sbn.swarmplot(
    data=data_final,
    x="site", y="auc", hue="Harmonization Schemes",
    order=site_order,
    hue_order=harm_modes, dodge=True, ax=ax
)
sbn.boxplot(
    data=data_final, color="w", zorder=1,
    x="site", y="auc", hue="Harmonization Schemes",
    order=site_order,
    hue_order=harm_modes, dodge=True, ax=ax, palette=["w"]*len(harm_modes)
)

handles, labels = ax.get_legend_handles_labels()
ax.legend(handles[:len(harm_modes)], labels[:len(harm_modes)])
plt.ylabel("AUC")
plt.xlabel("Sites")
plt.title("Diagnosis Classification")
plt.grid(alpha=0.5, axis="y", c="black")
plt.show()
table = table_generation(results_df)
print(table)
# %%
exp_dir = "/home/nnieto/Nico/Harmonization/results_classification/"
experiments_to_check = {
 'test_classification_Kersten_data_min30_logit_stack_gssvm_pred_5repetitions'
}
# Get results
results_df = extract_experiment_data(exp_dir, experiments_to_check, False)

data_final = get_fold_acc_auc(results_df)

# %%
# Change to appropiated names

data_final["Harmonization Schemes"].replace({"pretend": "JuHarmonize",
                                             "target": "Leakage",
                                             "none": "None",
                                             "cheat": "Cheat"},
                                            inplace=True)

# Select methods to plot
harm_modes = ["JuHarmonize", "Cheat", "Leakage", "None"]
data_final = data_final[data_final["Harmonization Schemes"].isin(harm_modes)]
data_final = data_final[data_final["auc"] != 0]
# Plot
site_order = ["Global", 1, 2, 130, 23, 73, 27, 41, 72, 127, 128]

# Plot
fig, ax = plt.subplots(1, 1, figsize=[20, 10])
sbn.swarmplot(
    data=data_final,
    x="site", y="balance_acc", hue="Harmonization Schemes",
    order=site_order,
    hue_order=harm_modes, dodge=True, ax=ax
)
sbn.boxplot(
    data=data_final, color="w", zorder=1,
    x="site", y="balance_acc", hue="Harmonization Schemes",
    order=site_order,
    hue_order=harm_modes, dodge=True, ax=ax, palette=["w"]*len(harm_modes)
)

handles, labels = ax.get_legend_handles_labels()
ax.legend(handles[:len(harm_modes)], labels[:len(harm_modes)])

plt.ylabel("Balanced Accuracy")
plt.xlabel("Sites ID")

site_list = ["Global",
             "S_1 [400]",
             "S_2 [157]",
             "S_130 [37]",
             "S_23 [36]",
             "S_73 [35]",
             "S_27 [34]",
             "S_41 [33]",
             "S_72 [33]",
             "S_127 [33]",
             "S_128 [32]",
             ]

ax.set_xticklabels(site_list)
plt.title("Diagnosis Classification")
plt.grid(alpha=0.5, axis="y", c="black")
plt.show()
table = table_generation(results_df)
print(table)

# %% REGRESSION KERSTEN DATA
# ###########################################################################################################################3
exp_dir = "/home/nnieto/Nico/Harmonization/results_regression/"
experiments_to_check = {
    'test_regression_Kersten_data_rf_stack_rvr_pred_5repetitions'
}
# Get results
data = extract_experiment_data(exp_dir, experiments_to_check, False)

data["harmonize_mode"].replace({"pretend": "JuHarmonize",
                                "target": "Leakage",
                                "none": "None",
                                "cheat": "Cheat"}, inplace=True)

harm_modes = ["JuHarmonize", "Cheat", "Leakage", "None"]

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

data["y_diff"] = np.abs(data["y_true"]-data["y_pred"])
sbn.catplot(
    data=data, kind="boxen",
    x="site", y="y_diff", hue="harmonize_mode",
    height=8, hue_order=harm_modes, legend_out=False
)
plt.ylabel("Age prediction difference [years]")
plt.title("Harmonization Schemes")
# sbn.move_legend(g, "upper right", bbox_to_anchor=(0.90, 0.98), frameon=False)
plt.title("Age Prediction")
table = table_generation(data)
print(table)
plt.grid(alpha=0.5, axis="y", c="black")
plt.show()
# %%
exp_dir = "/home/nnieto/Nico/Harmonization/results_regression/"
experiments_to_check = {
 'test_regression_Kersten_data_min30_rf_stack_rvr_pred_5repetitions'
}
# Get results
data = extract_experiment_data(exp_dir, experiments_to_check, False)

data["harmonize_mode"].replace({"pretend": "JuHarmonize",
                                "target": "Leakage",
                                "none": "None",
                                "cheat": "Cheat"}, inplace=True)

harm_modes = ["JuHarmonize", "Cheat", "Leakage", "None"]

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

data["y_diff"] = np.abs(data["y_true"]-data["y_pred"])

ax = sbn.catplot(
    data=data, kind="boxen",
    x="site", y="y_diff", hue="harmonize_mode",
    height=12, hue_order=harm_modes, legend_out=False
)
plt.ylabel("Age prediction difference [years]")
plt.title("Harmonization Schemes")
# sbn.move_legend(g, "upper right", bbox_to_anchor=(0.90, 0.98), frameon=False)
plt.title("Age Prediction")
table = table_generation(data)
print(table)
plt.grid(alpha=0.5, axis="y", c="black")

site_list = ["S_1 [400]",
             "S_2 [157]",
             "S_130 [37]",
             "S_23 [36]",
             "S_73 [35]",
             "S_27 [34]",
             "S_41 [33]",
             "S_72 [33]",
             "S_127 [33]",
             "S_128 [32]",
             ]

ax.set_xticklabels(site_list)
plt.title("Diagnosis Classification")
plt.grid(alpha=0.5, axis="y", c="black")
plt.show()


plt.show()



# %% OUT OF SAMPLE KERSTEN DATA CLASIFICATION 
# #######################################################################################################################################

exp_dir = "/home/nnieto/Nico/Harmonization/results_classification/results_classification_oos/"

for n, site in enumerate([1,2,130,23,73,27,41,72,127,128]):

    experiments_to_check = {'test_classification_oos_Kersten_data_SITE_'+str(site)+'_gssvm_pred_logit_stack'}
    results_exp_ID = extract_experiment_data_oos(exp_dir, experiments_to_check)
    results_exp_ID["site"] = "SITE_" + str(site)

    if n == 0:
        results_df = results_exp_ID
    else:
        results_df = pd.concat([results_df, results_exp_ID])

table = table_generation(results_df)
print(table)

get_fold_acc_auc
results_df["y_diff"] = (results_df["y_true"]-np.round(results_df["y_pred"]))
results_df["fold"] = 0
results_df["repeat"] = 1


data_final = get_fold_acc_auc(results_df)

# %%
# Change to appropiated names

data_final["Harmonization Schemes"].replace({"pretend_nosite": "JuHarmonize No site",
                                             "none": "None"},
                                            inplace=True)

# Select methods to plot
harm_modes = ["JuHarmonize No site", "None"]
data_final = data_final[data_final["Harmonization Schemes"].isin(harm_modes)]
# Plot

# Plot
fig, ax = plt.subplots(1, 1, figsize=[20, 10])

sbn.barplot(
    data=data_final, zorder=1,
    x="site", y="balance_acc", hue="Harmonization Schemes",
    hue_order=harm_modes, dodge=True, ax=ax
)

handles, labels = ax.get_legend_handles_labels()
ax.legend(handles[:len(harm_modes)], labels[:len(harm_modes)])
ax.axhline(0.5, lw=2, color="k", ls="--", alpha=1, label="Chance level")

plt.ylabel("Balanced Accuracy")
plt.xlabel("Sites ID")

site_list = ["Global",
             "S_1 [400]",
             "S_2 [157]",
             "S_130 [37]",
             "S_23 [36]",
             "S_73 [35]",
             "S_27 [34]",
             "S_41 [33]",
             "S_72 [33]",
             "S_127 [33]",
             "S_128 [32]",
             ]

ax.set_xticklabels(site_list)
plt.title("Diagnosis Classification - OOS Experiment")
plt.grid(alpha=0.5, axis="y", c="black")
plt.show()
table = table_generation(results_df)
print(table)

# %%
