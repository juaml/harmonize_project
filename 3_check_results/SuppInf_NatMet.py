# %%
import numpy as np
import pandas as pd
from pathlib import Path
import sys
import matplotlib.pyplot as plt
import seaborn as sbn

dir_path = '/home/nnieto/Nico/Harmonization/harmonize_project/3_check_results/'
__file__ = dir_path+'plot_NM.py'
to_append = Path(__file__).resolve().parent.parent.as_posix()
sys.path.append(to_append)

from lib.utils import extract_experiment_data, get_fold_acc_auc # noqa
from lib.utils import table_generation, extract_experiment_data # noqa
from lib.utils import extract_experiment_data_oos # noqa
# %% K features

SMALL_SIZE = 15
MEDIUM_SIZE = 20
BIGGER_SIZE = 30
plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

data = pd.read_csv(
    "/home/nnieto/Nico/Harmonization/results_kbest/"
    "last_experiment/collected/summaries.csv",
    sep=";")

data.rename(columns={"harmonize_mode": "Harmonization Schemes"},
            inplace=True)
data["Harmonization Schemes"].replace({"pretend": "JuHarmonize",
                                       "cheat": "Cheat"},
                                      inplace=True)

data_test = data[data["kind"] == "test"]
data_train = data[data["kind"] == "train"]
data_test["diff"] = data_test["mae"].to_numpy() - data_train["mae"].to_numpy()

_, ax = plt.subplots(1, 1, figsize=[20, 10])
pal = sbn.cubehelix_palette(2, rot=-.5, light=0.5, dark=0.2)

sbn.lineplot(data=data_test, x="k", y="diff", hue="Harmonization Schemes",
             ax=ax, palette=pal)
plt.xlim([data_test["k"].min(), data_test["k"].max()])
plt.grid()
plt.xlabel("Number of Features")
plt.ylabel("Test MAE - Train MAE")
plt.show()

# %% Number of images progression

number_used = [5, 10, 20, 50, 100, 150, 180, 200]
data_final = []
exp_dir = "/home/nnieto/Nico/Harmonization/results_regression/progression_experiment" # noqa
for n_ima in number_used:
    experiments_to_check = {'test_all_bigs_regression_stack_rf_pred_rvr_' + str(n_ima) + '_images'} # noqa
    data = extract_experiment_data(exp_dir, experiments_to_check)
    data["n_images"] = n_ima * 4
    print(n_ima)
    data_final.append(data)


data_final = pd.concat(data_final)

data_final["y_diff"] = (data_final["y_true"] -
                        data_final["y_pred"])

data_final["Harmonization Schemes"].replace({"pretend": "JuHarmonize",
                                      "target": "Leakage",
                                      "none": "None",
                                      "cheat": "Cheat"}, inplace=True)
harm_modes = ["JuHarmonize"]

data_final = data_final[data_final["Harmonization Schemes"].isin(harm_modes)]

_, ax = plt.subplots(1, 1, figsize=[20, 10])

sbn.boxplot(data=data_final, x="site", y="y_diff", hue="n_images",
            ax=ax)
plt.grid()
# %%
# %%

number_used = [20, 50, 100, 150, 180, 200, 230, 250, 300, 350]
data_final = []
exp_dir = "/home/nnieto/Nico/Harmonization/results_regression/progression_experiment" # noqa
for n_ima in number_used:
    experiments_to_check = {'test_all_bigs_regression_stack_rf_pred_rvr_' + str(n_ima) + '_images'} # noqa
    data = extract_experiment_data(exp_dir, experiments_to_check)
    data["n_images"] = n_ima
    print(n_ima)
    data_final.append(data)

data_final = pd.concat(data_final)

data_final["y_diff"] = (data_final["y_true"] -
                        data_final["y_pred"])

data_final["harmonize_mode"].replace({"pretend": "JuHarmonize",
                                      "target": "Leakage",
                                      "none": "None",
                                      "cheat": "Cheat"}, inplace=True)
harm_modes = ["JuHarmonize", "None"]

data_final = data_final[data_final["harmonize_mode"].isin(harm_modes)]

# sites_plot = ["ID1000"]

# data_final = data_final[data_final["site"].isin(sites_plot)]

_, ax = plt.subplots(1, 1, figsize=[20, 10])

sbn.boxplot(data=data_final, x="n_images", y="y_diff", hue="harmonize_mode",
            ax=ax)
plt.grid()
# %%
# %%

exp_dir = "/home/nnieto/Nico/Harmonization/results_regression/progression_experiment" # noqa
experiments_to_check = {'test_all_bigs_regression_stack_gsgpr_pred_gsgpr_all_images'} # noqa
data = extract_experiment_data_oos(exp_dir, experiments_to_check)


data["site"].replace({"1000Gehirne": "1000Brains",
                      "ID1000": "AOMIC-ID1000"}, inplace=True)


data["harmonize_mode"].replace({"pretend": "JuHarmonize",
                                "target": "Leakage",
                                "none": "None",
                                "cheat": "Cheat"}, inplace=True)

harm_modes = ["JuHarmonize", "Cheat", "Leakage", "None"]

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
# %% Separated by classifier

exp_dir = "/home/nnieto/Nico/Harmonization/results_regression/progression_experiment" # noqa
experiments_to_check = {'test_all_bigs_regression_stack_rvr_pred_rvr_all_images'} # noqa
data1 = extract_experiment_data_oos(exp_dir, experiments_to_check)
data1["stack_model"] = "rvr"
data1["pred_model"] = "rvr"
data1["models"] = "stack_rvr_pred_rvr"
print("RVR")

table = table_generation(data1)
print(table)
exp_dir = "/home/nnieto/Nico/Harmonization/results_regression/"
experiments_to_check = {'test_all_bigs_regression_stack_gsgpr_pred_gsgpr_all_images'} # noqa
data2 = extract_experiment_data_oos(exp_dir, experiments_to_check)
data2["stack_model"] = "gsgpr"
data2["pred_model"] = "gsgpr"
data2["models"] = "stack_gsgpr_pred_gsgpr"
print("GSGPR")
table = table_generation(data2)
print(table)
exp_dir = "/home/nnieto/Nico/Harmonization/results_r8"
experiments_to_check = {'test_all_regression_r8'}

data3 = extract_experiment_data(exp_dir, experiments_to_check)

data3["stack_model"] = "rf"
data3["pred_model"] = "rvr_original"
data3["models"] = "stack_rf_pred_rvr"

data = pd.concat([data1, data3, data2])

data["site"].replace({"1000Gehirne": "1000Brains",
                      "ID1000": "AOMIC-ID1000"}, inplace=True)


data["Harmonization Schemes"].replace({"pretend": "JuHarmonize",
                                "target": "Leakage",
                                "none": "None",
                                "cheat": "Cheat"}, inplace=True)

harm_modes = ["None"]
data = data[data["Harmonization Schemes"].isin(harm_modes)]


data["y_diff"] = (data["y_true"]-data["y_pred"])

pal = sbn.cubehelix_palette(4, rot=-.15, light=0.85, dark=0.3)
sbn.catplot(
    data=data, kind="boxen",
    x="site", y="y_diff", hue="pred_model",
    height=12, legend_out=False,

    palette=pal
)
plt.ylabel("Age prediction difference [years]")
plt.xlabel("Site Name")

plt.title("Harmonization Schemes")
# sbn.move_legend(g, "upper right", bbox_to_anchor=(0.90, 0.98), frameon=False)
plt.title("Age Prediction with None")
plt.grid(alpha=0.5, axis="y", c="black")
plt.show()
table = table_generation(data)
print(table)
# %%
# %%
exp_dir = "/home/nnieto/Nico/Harmonization/results_regression_oos/"
experiments_to_check = {'test_regression_use_site_CoRR_oos_all_bigs_pred_rvr_stack_rf_10img_min_corrected'} # noqa
data = extract_experiment_data_oos(exp_dir, experiments_to_check)
data.rename(columns={"harmonize_mode": "Harmonization Schemes"},
            inplace=True)

data["Harmonization Schemes"].replace(
                            {"pretend_nosite": "JuHarmonize",
                             "none": "None"},
                            inplace=True)


harm_modes = ["JuHarmonize", "None"]
absolute = False
site_list = ["Compleat DLBS", "DLBS"]
# Select methods to plot
data = data[data["Harmonization Schemes"].isin(harm_modes)]

data["site"].replace({"DLBS_full": "Compleat DLBS",
                     "DLBS": "Balanced DLBS"}, inplace=True)
# Plot
_, ax = plt.subplots(1, 1, figsize=[20, 10])

if absolute:
    data["y_diff"] = np.abs(data["y_true"] -
                            data["y_pred"])
else:
    data["y_diff"] = data["y_true"]-data["y_pred"]

pal = sbn.cubehelix_palette(2, rot=-.5, light=0.5, dark=0.2)
# sbn.swarmplot(
#     data=data, palette=pal,
#     x="site", y="y_diff", hue="Harmonization Schemes",
#     hue_order=harm_modes, dodge=True, ax=ax
# )
sbn.boxplot(
    data=data, color="w", zorder=1,
    x="site", y="y_diff", hue="Harmonization Schemes",
    hue_order=harm_modes, dodge=True, ax=ax, palette=pal
)
ax.axhline(0, lw=2, color="k", ls="-", alpha=1)
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles[:len(harm_modes)], labels[:len(harm_modes)])

plt.ylabel("Age Difference")
plt.xlabel("Sites ID")
plt.title("Age regression - OOS Experiment")
plt.grid(alpha=0.5, axis="y", c="black")
plt.show()

table = table_generation(data)
print(table)

# %%

# %%
