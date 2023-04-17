# %%
from pathlib import Path
import sys

dir_path = '/home/nnieto/Nico/Harmonization/harmonize_project/3_check_results/'
__file__ = dir_path+'plot_results_cv.py'
to_append = Path(__file__).resolve().parent.parent.as_posix()
sys.path.append(to_append)

from lib.utils import table_generation, plot_barplot, extract_experiment_data
from lib.utils import classification_table, plot_grup_barplot

# %%

exp_dir = "/home/nnieto/Nico/Harmonization/results_regression"
experiments_to_check = {
    'test_all_regression'
}

results_df = extract_experiment_data(exp_dir, experiments_to_check)
table = table_generation(results_df)
print(table)

plot_grup_barplot(results_df, True, True)
plot_grup_barplot(results_df, True, False, )

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
exp_dir = "/home/nnieto/Nico/Harmonization/results_regression/results_regression/"
experiments_to_check = {
        'test_all_data_regression_LinearSVR'
}

results_df = extract_experiment_data(exp_dir, experiments_to_check, False)
table = table_generation(results_df)

plot_barplot(results_df, True, False)
plot_grup_barplot(results_df, True, False, ["pretend", "cheat",
                                            "target", "none"])

# %%
exp_dir = "/home/nnieto/Nico/Harmonization/results_classification/"
experiments_to_check = {
        'test_classification_all_big_logit_stack'
}

results_df = extract_experiment_data(exp_dir, experiments_to_check, False)

classification_table(results_df, stats=["auc"])
# %%
exp_dir = "/home/nnieto/Nico/Harmonization/results_classification/"
experiments_to_check = {
        'test_classification_all_big_logit_stack_age_prediction_45yo'
}

results_df = extract_experiment_data(exp_dir, experiments_to_check, False)

classification_table(results_df, stats=["auc"])
# %%
exp_dir = "/home/nnieto/Nico/Harmonization/results_regression"
experiments_to_check = {
    'test_regression_all_big_rf_stack_rvr_pred'
}

results_df = extract_experiment_data(exp_dir, experiments_to_check)
table = table_generation(results_df)
print(table)

plot_grup_barplot(results_df, True, False, ["pretend", "none", "target", "cheat"])


# %%
# %%
exp_dir = "/home/nnieto/Nico/Harmonization/results_regression"
experiments_to_check = {
    'test_regression_all_bigs_all_aomic_rf_stack_rvr_pred'
}

results_df = extract_experiment_data(exp_dir, experiments_to_check)
table = table_generation(results_df)
print(table)

plot_grup_barplot(results_df, True, False, ["pretend", "none", "target", "cheat"])

# %%
