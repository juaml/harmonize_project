#%%
from pathlib import Path
import numpy as np
import pandas as pd

from pathlib import Path
import sys
__file__ = '/home/nnieto/Nico/Harmonization/harmonize_project/3_check_results/plot_results_oos.py'

to_append = Path(__file__).resolve().parent.parent.as_posix()
sys.path.append(to_append)

from lib.utils import table_generation, plot_barplot, extract_experiment_data, plot_barplot_classification, extract_experiment_data_oos
#%%
exp_dir = "/home/nnieto/Nico/Harmonization/results_r8"

experiments_to_check = {'test_all_big_oos_ID1000_r8'}
results_exp_ID = extract_experiment_data(exp_dir,experiments_to_check)
results_exp_ID["site"] = "ID1000"

experiments_to_check = {'test_all_big_oos_CamCAN_r8'}
results_exp_CamCAN = extract_experiment_data(exp_dir,experiments_to_check)
results_exp_CamCAN["site"] = "CamCAN"

experiments_to_check = {'test_all_big_oos_1000Gehirne_r8'}
results_exp_1000 = extract_experiment_data(exp_dir,experiments_to_check)
results_exp_1000["site"] = "1000Gehirne"

experiments_to_check = {'test_all_big_oos_eNKI_r8'}
results_exp_eNKI = extract_experiment_data(exp_dir,experiments_to_check)
results_exp_eNKI["site"] = "eNKI"
 
results_df = pd.concat([results_exp_ID,results_exp_CamCAN,results_exp_1000,results_exp_eNKI])

plot_barplot(results_df, False)
table = table_generation(results_df)
print(table)

#%%
import seaborn as sns
sns.color_palette("pastel")
import matplotlib.pyplot as plt

results_df["y_diff"] = results_df["y_true"]-results_df["y_pred"]

mask1 = results_df["harmonize_mode"] == "pretend_nosite" 
mask2 = results_df["harmonize_mode"] == "none" 

results_df_2plot = results_df[mask1+mask2]


g = sns.catplot(
    data=results_df_2plot, kind= "boxen",
    x="site", y="y_diff", hue="harmonize_mode", 
    height=6
)
g.set_axis_labels("", "Prediction difference")
g.legend.set_title("OOS experiment R8")
plt.grid(alpha=0.5,axis="y", c="black")
# %% CLASSIFICATION OOS

exp_dir = "/home/nnieto/Nico/Harmonization/results_classification/oos/"


experiments_to_check = {'test_all_big_oos_ID1000_classification'}
results_exp_ID = extract_experiment_data(exp_dir,experiments_to_check)
results_exp_ID["site"] = "ID1000"

experiments_to_check = {'test_all_big_oos_CamCAN_classification'}
results_exp_CamCAN = extract_experiment_data(exp_dir,experiments_to_check)
results_exp_CamCAN["site"] = "CamCAN"

experiments_to_check = {'test_all_big_oos_1000Gehirne_classification'}
results_exp_1000 = extract_experiment_data(exp_dir,experiments_to_check)
results_exp_1000["site"] = "1000Gehirne"

experiments_to_check = {'test_all_big_oos_eNKI_classification'}
results_exp_eNKI = extract_experiment_data(exp_dir,experiments_to_check)
results_exp_eNKI["site"] = "eNKI"

results_df = pd.concat([results_exp_CamCAN,results_exp_1000,results_exp_eNKI])

plot_barplot_classification(results_df, False)
table = table_generation(results_df, ["ACC"])
print(table)


# %% CLASSIFICATION OOS AGE
exp_dir = "/home/nnieto/Nico/Harmonization/results_classification/oos/"


experiments_to_check = {'test_all_big_oos_CamCAN_age_classification'}
results_exp_CamCAN = extract_experiment_data_oos(exp_dir,experiments_to_check)
results_exp_CamCAN["site"] = "CamCAN"

experiments_to_check = {'test_all_big_oos_1000Gehirne_age_classification'}
results_exp_1000 = extract_experiment_data_oos(exp_dir,experiments_to_check)
results_exp_1000["site"] = "1000Gehirne"

experiments_to_check = {'test_all_big_oos_eNKI_age_classification'}
results_exp_eNKI = extract_experiment_data_oos(exp_dir,experiments_to_check)
results_exp_eNKI["site"] = "eNKI"

results_df = pd.concat([results_exp_ID,results_exp_CamCAN,results_exp_1000,results_exp_eNKI])

plot_barplot_classification(results_df, False)
table = table_generation(results_df, ["ACC"])
print(table)

# %%import seaborn as sns

#%%
import seaborn as sns
sns.color_palette("pastel")
import matplotlib.pyplot as plt
results_df["y_diff"] = np.abs(results_df["y_true"]-results_df["y_pred"])

g = sns.catplot(
    data=results_df, kind= "boxen",
    x="site", y="y_diff", hue="harmonize_mode", 
    height=6
)
g.set_axis_labels("", "Prediction difference")
g.legend.set_title("OOS experiment R8")
plt.grid(alpha=0.5,axis="y", c="black")

# %%
