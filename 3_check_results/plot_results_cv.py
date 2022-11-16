#%%
import numpy as np
import pandas as pd
from pathlib import Path
import sys
__file__ = '/home/nnieto/Nico/Harmonization/harmonize_project/3_check_results/plot_results_cv.py'
to_append = Path(__file__).resolve().parent.parent.as_posix()
sys.path.append(to_append)

from lib.utils import table_generation, plot_barplot, extract_experiment_data , plot_grup_barplot
from lib.utils import classification_table
#%%
exp_dir = "/home/nnieto/Nico/Harmonization/results_r8"
experiments_to_check = {
    'test_all_regression_r8'
}

results_df = extract_experiment_data(exp_dir,experiments_to_check)
plot_grup_barplot(results_df, True, True)
plot_grup_barplot(results_df, True, False)

plot_barplot(results_df, True, False)
plot_barplot(results_df, True, True)

table = table_generation(results_df)
print(table)


# %%

exp_dir = "/home/nnieto/Nico/Harmonization/results"
experiments_to_check = {
    'test_all_regression'
}

results_df = extract_experiment_data(exp_dir,experiments_to_check)
table = table_generation(results_df)
print(table)



plot_grup_barplot(results_df, True, True)
plot_grup_barplot(results_df, True, False)

plot_barplot(results_df, True, False)
plot_barplot(results_df, True, True)

# %%


exp_dir = "/home/nnieto/Nico/Harmonization/results_classification"
experiments_to_check = {
    'test_all_classification'
}

results_df = extract_experiment_data(exp_dir,experiments_to_check)

classification_table(results_df)


# %%

exp_dir = "/home/nnieto/Nico/Harmonization/results_classification"
experiments_to_check = {
    'test_all_classification_RVC'
}

results_df = extract_experiment_data(exp_dir,experiments_to_check)

classification_table(results_df)
# %%

exp_dir = "/home/nnieto/Nico/Harmonization/results_classification"
experiments_to_check = {
    'test_all_classification_RVC_TIV_remove'
}

results_df = extract_experiment_data(exp_dir,experiments_to_check)

classification_table(results_df)
# %%

exp_dir = "/home/nnieto/Nico/Harmonization/results_classification"
experiments_to_check = {
    'test_all_classification_RVC_TIV_remove_25'
}

results_df = extract_experiment_data(exp_dir,experiments_to_check)

classification_table(results_df)
# %%
