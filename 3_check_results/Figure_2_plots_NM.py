# %%
import numpy as np
from pathlib import Path
import sys
from statannot.statannot import add_stat_annotation

import matplotlib.pyplot as plt
import seaborn as sbn

dir_path = '/home/nnieto/Nico/Harmonization/harmonize_project/3_check_results/'
__file__ = dir_path+'plot_NM.py'
to_append = Path(__file__).resolve().parent.parent.as_posix()
sys.path.append(to_append)

from lib.utils import extract_experiment_data, get_fold_acc_auc # noqa
from lib.utils import table_generation, classification_table # noqa


# %%
exp_dir = "/home/nnieto/Nico/Harmonization/results_r8"
experiments_to_check = {
    'test_all_regression_r8'
}
data = extract_experiment_data(exp_dir, experiments_to_check)
data["site"].replace({"1000Gehirne": "1000Brains",
                      "ID1000": "AOMIC-ID1000"}, inplace=True)
data["Harmonization Schemes"].replace({"pretend": "JuHarmonize",
                                       "target": "Leakage",
                                       "none": "None",
                                       "cheat": "Cheat",
                                       "predict": "NeuroHarmony",
                                       "notarget": "No Target"}, inplace=True)

harm_modes = ["JuHarmonize", "Cheat", "Leakage", "None", "No Target"]

data["y_diff"] = (data["y_true"]-data["y_pred"])

pal = sbn.cubehelix_palette(5, rot=-.15, light=0.85, dark=0.3)

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

# %%
# %% Creat a plot with the average
data["y_diff"] = np.abs(data["y_true"]-data["y_pred"])
data = data[data["Harmonization Schemes"].isin(harm_modes)]

fig = plt.figure(figsize=[10, 5])

ax = fig.add_subplot(1, 1, 1)
pal = sbn.cubehelix_palette(5, rot=-.15, light=0.85, dark=0.3)

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
plt.ylim([0, 6.3])

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
# %%
# %% Classification
exp_dir = "/home/nnieto/Nico/Harmonization/results_classification/"
experiments_to_check = {
    'test_classification_gender_all_bigs_logit_stack_gssvm_pred_5repetitions'
}
# Get results
data_final = extract_experiment_data(exp_dir, experiments_to_check, False)
data_final.rename(columns={"harmonize_mode": "Harmonization Schemes"},
                  inplace=True)

data_final = get_fold_acc_auc(data_final)

#
# Change to appropiated names
data_final["site"].replace({"1000Gehirne": "1000Brains",
                            "ID1000": "AOMIC-ID1000"}, inplace=True)

data_final["Harmonization Schemes"].replace({"pretend": "JuHarmonize",
                                             "target": "Leakage",
                                             "none": "None",
                                             "cheat": "Cheat",
                                             "notarget": "No Target"},
                                            inplace=True)

# Select methods to plot
harm_modes = ["JuHarmonize", "Cheat", "Leakage", "None", "No Target"]
data_final = data_final[data_final["Harmonization Schemes"].isin(harm_modes)]
site_order = ["Global", "AOMIC-ID1000", "eNKI", "1000Brains", "CamCAN"]
# Plot
fig, ax = plt.subplots(1, 1, figsize=[20, 10])
pal = sbn.cubehelix_palette(5, rot=-.5, light=0.5, dark=0.2)

sbn.swarmplot(
    data=data_final,
    x="site", y="auc", hue="Harmonization Schemes",
    order=site_order,
    hue_order=harm_modes, dodge=True, ax=ax,
    palette=pal
)

sbn.boxplot(
    data=data_final, color="w", zorder=1,
    x="site", y="auc", hue="Harmonization Schemes",
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

# %%
