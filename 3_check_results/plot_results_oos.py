# %%
from pathlib import Path
import numpy as np
import pandas as pd
import sys
import seaborn as sns
import matplotlib.pyplot as plt
__file__ = '/home/nnieto/Nico/Harmonization/harmonize_project/3_check_results/plot_results_oos.py'

to_append = Path(__file__).resolve().parent.parent.as_posix()
sys.path.append(to_append)
sns.color_palette("pastel")

from lib.utils import table_generation, plot_barplot, extract_experiment_data, plot_barplot_classification, extract_experiment_data_oos
# %%
exp_dir = "/home/nnieto/Nico/Harmonization/results_r8"

experiments_to_check = {'test_all_big_oos_ID1000_r8'}
results_exp_ID = extract_experiment_data(exp_dir, experiments_to_check)
results_exp_ID["site"] = "ID1000"

experiments_to_check = {'test_all_big_oos_CamCAN_r8'}
results_exp_CamCAN = extract_experiment_data(exp_dir, experiments_to_check)
results_exp_CamCAN["site"] = "CamCAN"

experiments_to_check = {'test_all_big_oos_1000Gehirne_r8'}
results_exp_1000 = extract_experiment_data(exp_dir, experiments_to_check)
results_exp_1000["site"] = "1000Gehirne"

experiments_to_check = {'test_all_big_oos_eNKI_r8'}
results_exp_eNKI = extract_experiment_data(exp_dir, experiments_to_check)
results_exp_eNKI["site"] = "eNKI"

results_df = pd.concat([results_exp_ID, results_exp_CamCAN,
                        results_exp_1000, results_exp_eNKI])

plot_barplot(results_df, False)
table = table_generation(results_df)
print(table)

# %%
results_df["y_diff"] = results_df["y_true"]-results_df["y_pred"]

mask1 = results_df["harmonize_mode"] == "pretend_nosite"
mask2 = results_df["harmonize_mode"] == "none"

results_df_2plot = results_df[mask1+mask2]


g = sns.catplot(
    data=results_df_2plot, kind="boxen",
    x="site", y="y_diff", hue="harmonize_mode",
    height=6
)
g.set_axis_labels("", "Prediction difference")
g.legend.set_title("OOS experiment R8")
plt.grid(alpha=0.5, axis="y", c="black")
# %% CLASSIFICATION OOS

exp_dir = "/home/nnieto/Nico/Harmonization/results_classification/oos/"


experiments_to_check = {'test_all_big_oos_ID1000_classification'}
results_exp_ID = extract_experiment_data(exp_dir, experiments_to_check)
results_exp_ID["site"] = "ID1000"

experiments_to_check = {'test_all_big_oos_CamCAN_classification'}
results_exp_CamCAN = extract_experiment_data(exp_dir, experiments_to_check)
results_exp_CamCAN["site"] = "CamCAN"

experiments_to_check = {'test_all_big_oos_1000Gehirne_classification'}
results_exp_1000 = extract_experiment_data(exp_dir, experiments_to_check)
results_exp_1000["site"] = "1000Gehirne"

experiments_to_check = {'test_all_big_oos_eNKI_classification'}
results_exp_eNKI = extract_experiment_data(exp_dir, experiments_to_check)
results_exp_eNKI["site"] = "eNKI"

results_df = pd.concat([results_exp_CamCAN, results_exp_1000, results_exp_eNKI])

plot_barplot_classification(results_df, False)
table = table_generation(results_df, ["ACC"])
print(table)


# %% CLASSIFICATION OOS AGE
exp_dir = "/home/nnieto/Nico/Harmonization/results_classification/oos/"


experiments_to_check = {'test_all_big_oos_CamCAN_age_classification'}
results_exp_CamCAN = extract_experiment_data_oos(exp_dir, experiments_to_check)
results_exp_CamCAN["site"] = "CamCAN"

experiments_to_check = {'test_all_big_oos_1000Gehirne_age_classification'}
results_exp_1000 = extract_experiment_data_oos(exp_dir, experiments_to_check)
results_exp_1000["site"] = "1000Gehirne"

experiments_to_check = {'test_all_big_oos_eNKI_age_classification'}
results_exp_eNKI = extract_experiment_data_oos(exp_dir, experiments_to_check)
results_exp_eNKI["site"] = "eNKI"

results_df = pd.concat([results_exp_ID, results_exp_CamCAN, results_exp_1000, results_exp_eNKI])

plot_barplot_classification(results_df, False)
table = table_generation(results_df, ["ACC"])
print(table)

# %%import seaborn as sns

#%%

results_df["y_diff"] = np.abs(results_df["y_true"]-results_df["y_pred"])

g = sns.catplot(
    data=results_df, kind="boxen",
    x="site", y="y_diff", hue="harmonize_mode", 
    height=6
)
g.set_axis_labels("", "Prediction difference")
g.legend.set_title("OOS experiment R8")
plt.grid(alpha=0.5, axis="y", c="black")

# %%

exp_dir = "/home/nnieto/Nico/Harmonization/result_regression/results_regression/"

experiments_to_check = {'test_all_big_oos_ID1000_rf_stack_rvr_pred_all_aomic'}
results_exp_ID = extract_experiment_data(exp_dir, experiments_to_check)
results_exp_ID["site"] = "ID1000"

experiments_to_check = {'test_all_big_oos_CamCAN_rf_stack_rvr_pred_all_aomic'}
results_exp_CamCAN = extract_experiment_data(exp_dir, experiments_to_check)
results_exp_CamCAN["site"] = "CamCAN"

experiments_to_check = {'test_all_big_oos_1000Gehirne_rf_stack_rvr_pred_all_aomic'}
results_exp_1000 = extract_experiment_data(exp_dir, experiments_to_check)
results_exp_1000["site"] = "1000Gehirne"

experiments_to_check = {'test_all_big_oos_eNKI_rf_stack_rvr_pred_all_aomic'}
results_exp_eNKI = extract_experiment_data(exp_dir, experiments_to_check)
results_exp_eNKI["site"] = "eNKI"

experiments_to_check = {'test_all_big_oos_PIOP1_rf_stack_rvr_pred_all_aomic'}
results_exp_PIOP1 = extract_experiment_data(exp_dir, experiments_to_check)
results_exp_PIOP1["site"] = "PIOP1"

experiments_to_check = {'test_all_big_oos_PIOP2_rf_stack_rvr_pred_all_aomic'}
results_exp_PIOP2 = extract_experiment_data(exp_dir, experiments_to_check)
results_exp_PIOP2["site"] = "PIOP2"

results_df = pd.concat([results_exp_ID, results_exp_CamCAN, results_exp_1000,
                        results_exp_eNKI, results_exp_PIOP1, results_exp_PIOP2])

table = table_generation(results_df)
print(table)

results_df["y_diff"] = (results_df["y_true"]-np.round(results_df["y_pred"]))

# %%
data_final = results_df
data_final.rename(columns={"harmonize_mode": "Harmonization Schemes"},
                  inplace=True)
# Change to appropiated names
data_final["site"].replace({"1000Gehirne": "1000Brains",
                            "ID1000": "AOMIC-ID1000",
                            "PIOP1": "AOMIC-PIOP1",
                            "PIOP2": "AOMIC-PIOP2"}, inplace=True)

data_final["Harmonization Schemes"].replace({"pretend_nosite": "JuHarmonize",
                                             "predict": "Predict Model",
                                             "none": "None",
                                             "predict_pretend_nosite": "Predict No Site"},
                                            inplace=True)

# Select methods to plot
harm_modes = ["JuHarmonize", "Predict Model", "Predict No Site", "None"]
data_final = data_final[data_final["Harmonization Schemes"].isin(harm_modes)]

plt.figure(figsize=[20, 10])
sns.catplot(
    data=data_final, kind="boxen", hue_order=harm_modes,
    x="site", y="y_diff", hue="Harmonization Schemes",
    order=["AOMIC-ID1000", "AOMIC-PIOP1", "AOMIC-PIOP2",
           "eNKI", "1000Brains", "CamCAN"],
    height=10,
)
plt.ylabel("Absolute Prediction difference [years]")
plt.title("Out of Sample Age Prediction")
plt.xlabel("Sites")
plt.grid(alpha=0.5, axis="y", c="black")

# %%
