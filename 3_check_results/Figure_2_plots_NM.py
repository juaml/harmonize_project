# %%
import numpy as np
import pandas as pd
from pathlib import Path
import sys
from statannot.statannot import add_stat_annotation

import matplotlib.pyplot as plt
import seaborn as sbn
from sklearn.metrics import (
    mean_absolute_error
)
dir_path = '/home/nnieto/Nico/Harmonization/harmonize_project/3_check_results/'
__file__ = dir_path+'plot_NM.py'
to_append = Path(__file__).resolve().parent.parent.as_posix()
sys.path.append(to_append)

from lib.utils import extract_experiment_data, get_fold_acc_auc # noqa
from lib.utils import table_generation # noqa


# %%
exp_dir = "/home/nnieto/Nico/Harmonization/results_r8"
experiments_to_check = {
    'test_all_regression_r8'
}

data = extract_experiment_data(exp_dir, experiments_to_check)


data["site"].replace({"1000Gehirne": "1000Brains",
                      "ID1000": "AOMIC-ID1000"}, inplace=True)


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
sort_mode = df["method"]

data["y_diff"] = (data["y_true"]-data["y_pred"])

data.rename(columns={"harmonize_mode": "Harmonization Schemes"},
            inplace=True)

pal = sbn.cubehelix_palette(4, rot=-.15, light=0.85, dark=0.3)
sbn.catplot(
    data=data, kind="boxen",
    x="site", y="y_diff", hue="Harmonization Schemes",
    height=12, hue_order=sort_mode, legend_out=False,
    order=["AOMIC-ID1000", "eNKI", "1000Brains", "CamCAN"],
    palette=pal
)
plt.ylabel("Age prediction difference [years]")
plt.xlabel("Site Name")

plt.title("Harmonization Schemes")
# sbn.move_legend(g, "upper right", bbox_to_anchor=(0.90, 0.98), frameon=False)
plt.title("Age Prediction")
plt.grid(alpha=0.5, axis="y", c="black")
plt.show()
table = table_generation(data)
print(table)

# %% Creat a plot with the average
data["y_diff"] = np.abs(data["y_true"]-data["y_pred"])
data = data[data["Harmonization Schemes"].isin(harm_modes)]

fig = plt.figure(figsize=[10, 5])

ax = fig.add_subplot(1, 1, 1)
pal = sbn.cubehelix_palette(4, rot=-.15, light=0.85, dark=0.3)

ax = sbn.barplot(
    data=data,
    x="Harmonization Schemes", y="y_diff",
    order=harm_modes, ax=ax,
    palette=pal, seed=23, n_boot=1000
)
plt.ylabel("Absolute age prediction difference [years]")
plt.xlabel("Harmonization Schemes")
# sbn.move_legend(g, "upper right", bbox_to_anchor=(0.90, 0.98), frameon=False)
table = table_generation(data)
print(table)
plt.grid(alpha=0.5, axis="y", c="black")
plt.ylim([0, 5.5])

# Comparisons for statistical test
box_list = [("JuHarmonize", "Cheat"),
            ("JuHarmonize", "Leakage"),
            ("JuHarmonize", "None")]

add_stat_annotation(ax, data=data, x="Harmonization Schemes", y="y_diff",
                    box_pairs=box_list, test='Wilcoxon',
                    text_format='star', loc='outside', order=harm_modes,
                    verbose=1,
                    pvalue_thresholds=[[1, "ns"], [0.01, "* p<0.01"]])
plt.show()

# %% Classification

exp_dir = "/home/nnieto/Nico/Harmonization/result_classification/"
experiments_to_check = {
    'test_classification_gender_all_bigs_logit_stack_gssvm_pred_5repetitions'
}

# Get results
results_df = extract_experiment_data(exp_dir, experiments_to_check, False)
data_final = get_fold_acc_auc(results_df)

#
# Change to appropiated names
data_final["site"].replace({"1000Gehirne": "1000Brains",
                            "ID1000": "AOMIC-ID1000"}, inplace=True)

data_final["Harmonization Schemes"].replace({"pretend": "JuHarmonize",
                                             "target": "Leakage",
                                             "none": "None",
                                             "cheat": "Cheat"},
                                            inplace=True)

# Select methods to plot
harm_modes = ["JuHarmonize", "Cheat", "Leakage", "None"]
data_final = data_final[data_final["Harmonization Schemes"].isin(harm_modes)]
site_order = ["Global", "AOMIC-ID1000", "eNKI", "1000Brains", "CamCAN"]
# Plot
fig, ax = plt.subplots(1, 1, figsize=[20, 10])
pal = sbn.cubehelix_palette(4, rot=-.5, light=0.5, dark=0.2)

sbn.swarmplot(
    data=data_final,
    x="site", y="acc", hue="Harmonization Schemes",
    order=site_order,
    hue_order=harm_modes, dodge=True, ax=ax,
    palette=pal
)

sbn.boxplot(
    data=data_final, color="w", zorder=1,
    x="site", y="acc", hue="Harmonization Schemes",
    order=site_order,
    hue_order=harm_modes, dodge=True, ax=ax, palette=["w"]*len(harm_modes)
)

handles, labels = ax.get_legend_handles_labels()
ax.legend(handles[:len(harm_modes)], labels[:len(harm_modes)])
ax.set_ylim([0.85, 0.99])

plt.ylabel("Balanced Accuracy")
plt.xlabel("Sites")
plt.title("Gender Classification")
plt.grid(alpha=0.5, axis="y", c="black")
plt.show()
results_df.rename(columns={"harmonize_mode": "Harmonization Schemes"},
                  inplace=True)
table = table_generation(results_df)
print(table)

# %%
exp_dir = "/home/nnieto/Nico/Harmonization/results_classification/"
experiments_to_check = {
    'test_classification_all_bigs_logit_stack_gssvm_pred_5repetitions_TIV20'
}
# Get results
results_df = extract_experiment_data(exp_dir, experiments_to_check, False)
data_final = get_fold_acc_auc(results_df)

# Change to appropiated names
data_final["site"].replace({"1000Gehirne": "1000Brains",
                            "ID1000": "AOMIC-ID1000"}, inplace=True)

data_final["Harmonization Schemes"].replace({"pretend": "JuHarmonize",
                                             "target": "Leakage",
                                             "none": "None",
                                             "cheat": "Cheat"},
                                            inplace=True)

# Select methods to plot
harm_modes = ["JuHarmonize", "Cheat", "Leakage", "None"]
data_final = data_final[data_final["Harmonization Schemes"].isin(harm_modes)]
site_order = ["Global", "AOMIC-ID1000", "eNKI", "1000Brains", "CamCAN"]
# Plot
fig, ax = plt.subplots(1, 1, figsize=[20, 10])

sbn.swarmplot(
    data=data_final,
    x="site", y="acc", hue="Harmonization Schemes",
    order=site_order, palette=pal,
    hue_order=harm_modes, dodge=True, ax=ax
)

sbn.boxplot(
    data=data_final, color="w", zorder=1,
    x="site", y="acc", hue="Harmonization Schemes",
    order=site_order,
    hue_order=harm_modes, dodge=True, ax=ax, palette=["w"]*len(harm_modes)
)

handles, labels = ax.get_legend_handles_labels()
ax.legend(handles[:len(harm_modes)], labels[:len(harm_modes)])
ax.set_ylim([0.85, 0.99])

plt.ylabel("Balanced Accuracy")
plt.xlabel("Sites")
plt.title("Gender Classification 20% TIV Removal")
plt.grid(alpha=0.5, axis="y", c="black")
plt.show()
results_df.rename(columns={"harmonize_mode": "Harmonization Schemes"},
                  inplace=True)
table = table_generation(results_df)
print(table)
# %%
exp_dir = "/home/nnieto/Nico/Harmonization/results_r8"
experiments_to_check = {
    'test_all_regression_r8'
}

data = extract_experiment_data(exp_dir, experiments_to_check)


data["site"].replace({"1000Gehirne": "1000Brains",
                      "ID1000": "AOMIC-ID1000"}, inplace=True)


data["harmonize_mode"].replace({"pretend": "JuHarmonize",
                                "target": "Passing Age as Covar",
                                "none": "Just Pooling data",
                                "cheat": "Cheat",
                                "notarget": "NOT Passing Age as Covar"},
                               inplace=True)

harm_modes = ["Passing Age as Covar", "NOT Passing Age as Covar"]

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

data["y_diff"] = (data["y_true"]-data["y_pred"])

data.rename(columns={"harmonize_mode": "Harmonization Schemes"},
            inplace=True)

pal = sbn.cubehelix_palette(4, rot=-.15, light=0.85, dark=0.3)
sbn.catplot(
    data=data, kind="boxen",
    x="site", y="y_diff", hue="Harmonization Schemes",
    height=12, hue_order=sort_mode, legend_out=False,
    order=["AOMIC-ID1000", "eNKI", "1000Brains", "CamCAN"],
    palette=pal
)
plt.ylabel("Age prediction difference [years]")
plt.xlabel("Site Name")

plt.title("Harmonization Schemes")
# sbn.move_legend(g, "upper right", bbox_to_anchor=(0.90, 0.98), frameon=False)
plt.title("Age Prediction")
plt.grid(alpha=0.5, axis="y", c="black")
plt.show()
# %%
data["y_diff"] = np.abs(data["y_true"]-data["y_pred"])
data = data[data["Harmonization Schemes"].isin(harm_modes)]

fig = plt.figure(figsize=[10, 5])

ax = fig.add_subplot(1, 1, 1)
pal = sbn.cubehelix_palette(4, rot=-.15, light=0.85, dark=0.3)

ax = sbn.barplot(
    data=data,
    x="Harmonization Schemes", y="y_diff",
    order=harm_modes, ax=ax,
    palette=pal, seed=23, n_boot=1000
)
plt.ylabel("Absolute age prediction difference [years]")
plt.xlabel("Harmonization Schemes")
# sbn.move_legend(g, "upper right", bbox_to_anchor=(0.90, 0.98), frameon=False)
table = table_generation(data)
print(table)
plt.grid(alpha=0.5, axis="y", c="black")
# plt.ylim([0, 5.5])

plt.show()

# %%


exp_dir = "/home/nnieto/Nico/Harmonization/results_regression/"
experiments_to_check = {
    'test_all_bigs_regression_stack_gsgpr_pred_gsgpr_all_images'
}

data = extract_experiment_data(exp_dir, experiments_to_check)


data["site"].replace({"1000Gehirne": "1000Brains",
                      "ID1000": "AOMIC-ID1000"}, inplace=True)


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
sort_mode = df["method"]

data["y_diff"] = (data["y_true"]-data["y_pred"])

data.rename(columns={"harmonize_mode": "Harmonization Schemes"},
            inplace=True)

pal = sbn.cubehelix_palette(4, rot=-.15, light=0.85, dark=0.3)
sbn.catplot(
    data=data, kind="boxen",
    x="site", y="y_diff", hue="Harmonization Schemes",
    height=12, hue_order=harm_modes, legend_out=False,
    order=["AOMIC-ID1000", "eNKI", "1000Brains", "CamCAN"],
    palette=pal
)
plt.ylabel("Age prediction difference [years]")
plt.xlabel("Site Name")

plt.title("Harmonization Schemes")
# sbn.move_legend(g, "upper right", bbox_to_anchor=(0.90, 0.98), frameon=False)
plt.title("Age Prediction")
plt.grid(alpha=0.5, axis="y", c="black")
plt.show()
table = table_generation(data)
print(table)
