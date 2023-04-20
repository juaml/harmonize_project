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
    "/home/nnieto/Nico/Harmonization/results_kbest/last_experiment/collected/summaries.csv",
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
# %% Kersten Data, Alzheimer dementia classification. 2 sites #################
exp_dir = "/home/nnieto/Nico/Harmonization/results_classification/"
experiments_to_check = {
    'test_classification_Kersten_data_logit_stack_gssvm_pred_5repetitions'
}
# Get results
data = extract_experiment_data(exp_dir, experiments_to_check, False)
data.rename(columns={"harmonize_mode": "Harmonization Schemes"},
            inplace=True)
data_final = get_fold_acc_auc(data)
# Change to appropiated names
data["Harmonization Schemes"].replace({"pretend": "JuHarmonize",
                                       "target": "Leakage",
                                       "none": "None",
                                       "cheat": "Cheat"},
                                      inplace=True)

# Select methods to plot
harm_modes = ["JuHarmonize", "Leakage", "Cheat", "None"]
data = data[data["Harmonization Schemes"].isin(harm_modes)]
site_order = ["Global", "SITE_1", "SITE_2"]

# Plot
pal = sbn.cubehelix_palette(4, rot=-.5, light=0.5, dark=0.2)
_, ax = plt.subplots(1, 1, figsize=[20, 10])
sbn.swarmplot(
    data=data,
    x="site", y="acc", hue="Harmonization Schemes",
    order=site_order, palette=pal,
    hue_order=harm_modes, dodge=True, ax=ax
)
sbn.boxplot(
    data=data, color="w", zorder=1,
    x="site", y="acc", hue="Harmonization Schemes",
    order=site_order,
    hue_order=harm_modes, dodge=True, ax=ax, palette=["w"]*len(harm_modes)
)

handles, labels = ax.get_legend_handles_labels()
ax.legend(handles[:len(harm_modes)], labels[:len(harm_modes)],
          loc='lower left')
plt.ylabel("Balanced Accuracy")
plt.xlabel("Sites")
plt.title("Diagnosis Classification")
plt.grid(alpha=0.5, axis="y", c="black")
plt.show()

table = table_generation(data)
print(table)

# %%
exp_dir = "/home/nnieto/Nico/Harmonization/results_classification/"
experiments_to_check = {
 'test_classification_Kersten_data_min30_logit_stack_gssvm_pred_5repetitions'
}
# Get results
data = extract_experiment_data(exp_dir, experiments_to_check, False)
data.rename(columns={"harmonize_mode": "Harmonization Schemes"},
            inplace=True)
data = get_fold_acc_auc(data)

data.rename(columns={"harmonize_mode": "Harmonization Schemes"},
            inplace=True)
data["Harmonization Schemes"].replace({"pretend": "JuHarmonize",
                                       "target": "Leakage",
                                       "none": "None",
                                       "cheat": "Cheat"}, inplace=True)

harm_modes = ["JuHarmonize", "Cheat", "Leakage", "None"]
# Select methods to plot
data = data[data["Harmonization Schemes"].isin(harm_modes)]

pal = sbn.cubehelix_palette(4, rot=-.5, light=0.5, dark=0.2)
_, ax = plt.subplots(1, 1, figsize=[20, 10])

sbn.boxplot(
    data=data, color="w",
    x="site", y="balance_acc", hue="Harmonization Schemes",
    hue_order=harm_modes, dodge=True, ax=ax, palette=pal
)

plt.ylabel("Balanced Accuracy")
plt.xlabel("Sites")
plt.title("Diagnosis Classification")
plt.grid(alpha=0.5, axis="y", c="black")

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
plt.show()

# %% OUT OF SAMPLE KERSTEN DATA CLASIFICATION
# ###########################################################################
exp_dir = "/home/nnieto/Nico/Harmonization/results_classification/results_classification_oos/"

for n, site in enumerate([1, 2, 130, 23, 73, 27, 41, 72, 127, 128]):
    to_check = {'test_classification_oos_Kersten_data_SITE_'+str(site)+'_gssvm_pred_logit_stack'}
    results_exp_ID = extract_experiment_data_oos(exp_dir, to_check)
    results_exp_ID["site"] = "SITE_" + str(site)

    if n == 0:
        data = results_exp_ID
    else:
        data = pd.concat([data, results_exp_ID])

data.rename(columns={"harmonize_mode": "Harmonization Schemes"},
            inplace=True)
table = table_generation(data)
print(table)

data["y_diff"] = (data["y_true"]-np.round(data["y_pred"]))
data["fold"] = 0
data["repeat"] = 1

data = get_fold_acc_auc(data)

# Change to appropiated names
data["Harmonization Schemes"].replace({"pretend_nosite":
                                       "JuHarmonize",
                                       "none": "None"},
                                      inplace=True)

# Select methods to plot
harm_modes = ["JuHarmonize", "None"]
data = data[data["Harmonization Schemes"].isin(harm_modes)]
# Plot

# Plot
fig, ax = plt.subplots(1, 1, figsize=[20, 10])
pal = sbn.cubehelix_palette(2, rot=-.5, light=0.5, dark=0.2)

sbn.barplot(
    data=data, zorder=1, palette=pal,
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
table = table_generation(data)
print(table)

# %% BALANCED DATA ############################################################
exp_dir = "/home/nnieto/Nico/Harmonization/results_regression/"
experiments_to_check = {'test_regression_balanced_data_rf_stack_rvr_pred'}
data = extract_experiment_data(exp_dir, experiments_to_check, train_acc=False)
data.rename(columns={"harmonize_mode": "Harmonization Schemes"},
            inplace=True)


data["Harmonization Schemes"].replace({"pretend": "JuHarmonize",
                                       "target": "Leakage",
                                       "none": "None",
                                       "cheat": "Cheat"}, inplace=True)
harm_modes = ["JuHarmonize", "Cheat", "Leakage", "None"]
data = data[data["Harmonization Schemes"].isin(harm_modes)]

absolute = False
if absolute:
    data["y_diff"] = np.abs(data["y_true"]-data["y_pred"])
else:
    data["y_diff"] = data["y_true"]-data["y_pred"]

fig, ax = plt.subplots(1, 1, figsize=[20, 10])
pal = sbn.cubehelix_palette(4, rot=-.5, light=0.5, dark=0.2)

sbn.boxenplot(
    data=data, palette=pal,
    x="site", y="y_diff", hue="Harmonization Schemes",
    hue_order=harm_modes
)
plt.ylabel("Age prediction difference [years]")
plt.title("Age Prediction")
plt.xlabel("Sites ID")

plt.grid(alpha=0.5, axis="y", c="black")
plt.show()

table = table_generation(data)
print(table)
# %%
exp_dir = "/home/nnieto/Nico/Harmonization/results_regression_oos/"
experiments_to_check = {'test_regression_oos_balanced_data_DLBS_full_rvr_pred_rvr_stack'}
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
sbn.swarmplot(
    data=data, palette=pal,
    x="site", y="y_diff", hue="Harmonization Schemes",
    hue_order=harm_modes, dodge=True, ax=ax
)
sbn.boxplot(
    data=data, color="w", zorder=1,
    x="site", y="y_diff", hue="Harmonization Schemes",
    hue_order=harm_modes, dodge=True, ax=ax, palette=["w"]*len(harm_modes)
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

# %% Number of images progression

number_used = [50, 100, 150, 200, 250, 300, 350]
data_final = []
exp_dir = "/home/nnieto/Nico/Harmonization/results_regression/progression_experiment"
for n_ima in number_used:
    experiments_to_check = {'test_all_bigs_regression_stack_gsgpr_pred_gsgpr_' + str(n_ima) + '_images'}
    data = extract_experiment_data_oos(exp_dir, experiments_to_check)
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

sites_plot = ["ID1000"]

data_final = data_final[data_final["site"].isin(sites_plot)]

_, ax = plt.subplots(1, 1, figsize=[20, 10])

sbn.boxplot(data=data_final, x="n_images", y="y_diff", hue="harmonize_mode",
            ax=ax, hue_order=harm_modes)
plt.grid()
# %%
# %%

exp_dir = "/home/nnieto/Nico/Harmonization/results_regression/progression_experiment"
experiments_to_check = {'test_all_bigs_regression_stack_gsgpr_pred_gsgpr_all_images'}
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

exp_dir = "/home/nnieto/Nico/Harmonization/results_regression/progression_experiment"
experiments_to_check = {'test_all_bigs_regression_stack_rvr_pred_rvr_all_images'}
data1 = extract_experiment_data_oos(exp_dir, experiments_to_check)
data1["stack_model"] = "rvr"
data1["pred_model"] = "rvr"
data1["models"] = "stack_rvr_pred_rvr"

exp_dir = "/home/nnieto/Nico/Harmonization/results_regression/"
experiments_to_check = {'test_all_bigs_regression_stack_gsgpr_pred_gsgpr_all_images'}
data2 = extract_experiment_data_oos(exp_dir, experiments_to_check)
data2["stack_model"] = "gsgpr"
data2["pred_model"] = "gsgpr"
data2["models"] = "stack_gsgpr_pred_gsgpr"

exp_dir = "/home/nnieto/Nico/Harmonization/results_r8"
experiments_to_check = {'test_all_regression_r8'}

data3 = extract_experiment_data(exp_dir, experiments_to_check)

data3["stack_model"] = "rf"
data3["pred_model"] = "rvr_original"
data3["models"] = "stack_rf_pred_rvr"

data = pd.concat([data1, data3, data2])

data["site"].replace({"1000Gehirne": "1000Brains",
                      "ID1000": "AOMIC-ID1000"}, inplace=True)


data["harmonize_mode"].replace({"pretend": "JuHarmonize",
                                "target": "Leakage",
                                "none": "None",
                                "cheat": "Cheat"}, inplace=True)

harm_modes = ["None"]
data = data[data["harmonize_mode"].isin(harm_modes)]


data["y_diff"] = (data["y_true"]-data["y_pred"])

data.rename(columns={"harmonize_mode": "Harmonization Schemes"},
            inplace=True)

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
