# %%
import numpy as np
import pandas as pd
import pickle
from juharmonize import JuHarmonize, JuHarmonizeClassifier
from juharmonize.utils import subset_data
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score,
    roc_auc_score,
)
# Get the index and filter the data


def compute_metric(model, X, y, predicted_y, acc, auc_score):
    predicted_y_loop = model.predict_proba(X)
    acc_loop = balanced_accuracy_score(y, np.round(predicted_y_loop[:, 1]))
    auc_score_loop = roc_auc_score(y, predicted_y_loop[:, 1])
    predicted_y.append(predicted_y_loop)
    acc = np.append(acc, acc_loop)
    auc_score = np.append(auc_score, auc_score_loop)
    return predicted_y, acc, auc_score


def compute_metric_pretend(model, X, y, sites, predicted_y,
                           acc, auc_score):
    predicted_y_loop = model.predict_proba(X, sites=sites)
    acc_loop = balanced_accuracy_score(y, np.round(predicted_y_loop[:, 1]))
    auc_score_loop = roc_auc_score(y, predicted_y_loop[:, 1])
    predicted_y.append(predicted_y_loop)
    acc = np.append(acc, acc_loop)
    auc_score = np.append(auc_score, auc_score_loop)
    return predicted_y, acc, auc_score


# WM
save_dir = "/home/nnieto/Nico/Harmonization/data/eICU/Results/10_images_min/"

root_dir = "/home/nnieto/Nico/Harmonization/data/eICU/"
data = pd.read_csv(root_dir + "equals_to_paper_data.csv", index_col=0)
ABG_of_interes = ["paO2", "paCO2", "pH", "Base Excess",
                  "Hgb", "glucose", "bicarbonate", "lactate"]

min_images_per_site = 10
# Remove sites with less than a thd number of patients
site_counts = data["site"].value_counts()

# filter the site_ids with less than a thd
mask = site_counts[site_counts > min_images_per_site].index.tolist()

# Filter the sites with the minimun number of patietes
data = data[data['site'].isin(mask)]

#
X = data.loc[:, ABG_of_interes].to_numpy()
sites = data["site"].to_numpy()
Y = data["endpoint"].replace({"Expired": 1, "Alive": 0}).to_numpy()

scaler = RobustScaler()

pred_model = LogisticRegression()

stack_model = LogisticRegression()

cheat_model = JuHarmonize()
X = scaler.fit_transform(X, Y)
X_cheat = cheat_model.fit_transform(X, Y, sites)
kf = RepeatedStratifiedKFold(n_splits=3, n_repeats=10,
                             random_state=43)

# %%
pred_target = []
pred_notarget = []
pred_cheat = []
pred_none = []
pred_target_trn = []
pred_notarget_trn = []
pred_cheat_trn = []
pred_none_trn = []

acc_none = []
acc_target = []
acc_notarget = []
acc_cheat = []

auc_none = []
auc_target = []
auc_notarget = []
auc_cheat = []

acc_none_trn = []
acc_target_trn = []
acc_notarget_trn = []
acc_cheat_trn = []

auc_none_trn = []
auc_target_trn = []
auc_notarget_trn = []
auc_cheat_trn = []

pred_pretend = []
acc_pretend = []
auc_pretend = []

pred_pretend_trn = []
acc_pretend_trn = []
auc_pretend_trn = []

y_test_list = []
y_train_list = []
sites_train_list = []
sites_test_list = []
folds_lits = []
#
for i_fold, (train_index, test_index) in enumerate(kf.split(X, Y,
                                                            groups=sites)):
    print("FOLD: " + str(i_fold))

    X_train, sites_train, y_train, covars_train, _ = subset_data(
        train_index, X, sites, Y
    )

    X_test, sites_test, y_test, covars_test, _ = subset_data(
        test_index, X, sites, Y)

    y_test_list.append(y_test)
    y_train_list.append(y_train)
    sites_train_list.append(sites_train)
    sites_test_list.append(sites_test)
    folds_lits.append(i_fold*np.ones(y_test.shape))
    # # pretend
    harm_model = JuHarmonizeClassifier(pred_model=pred_model,
                                       stack_model=stack_model,
                                       use_cv_test_transforms=True)
    harm_model.fit(X_train, y_train, sites_train)

    pred_pretend, acc_pretend, auc_pretend = \
        compute_metric_pretend(harm_model, X_test, y_test,
                               sites_test, pred_pretend,
                               acc_pretend, auc_pretend,
                               )

    pred_pretend_trn, acc_pretend_trn, auc_pretend_trn = \
        compute_metric_pretend(harm_model, X_train, y_train,
                               sites_train, pred_pretend_trn,
                               acc_pretend_trn, auc_pretend_trn,
                               )

    # none
    pred_model.fit(X_train, y_train)
    pred_none, acc_none, auc_none = \
        compute_metric(pred_model, X_test, y_test,
                       pred_none, acc_none, auc_none)

    pred_none_trn, acc_none_trn, auc_none_trn = \
        compute_metric(pred_model, X_train, y_train,
                       pred_none_trn, acc_none_trn, auc_none_trn)

    # target
    target_model = JuHarmonize()
    X_harm = target_model.fit_transform(X_train, y_train, sites_train)
    pred_model.fit(X_harm, y_train)
    X_test_harm = target_model.transform(X_test, y_test, sites_test)
    pred_target, acc_target, auc_target = \
        compute_metric(pred_model, X_test, y_test,
                       pred_target, acc_target, auc_target)
    pred_target_trn, acc_target_trn, auc_target_trn = \
        compute_metric(pred_model, X_train, y_train,
                       pred_target_trn, acc_target_trn, auc_target_trn,
                       )

    # no target
    harm_model = JuHarmonize(preserve_target=False)
    X_harm = harm_model.fit_transform(X_train, y_train, sites_train)
    pred_model.fit(X_harm, y_train)
    X_test_harm = harm_model.transform(X_test, y_test, sites_test)
    pred_notarget, acc_notarget, auc_notarget = \
        compute_metric(pred_model, X_test, y_test,
                       pred_notarget, acc_notarget, auc_notarget,
                       )

    pred_notarget_trn, acc_notarget_trn, auc_notarget_trn = \
        compute_metric(pred_model, X_train, y_train,
                       pred_notarget_trn, acc_notarget_trn, auc_notarget_trn,
                       )

    # Cheat
    X_train, sites_train, y_train, covars_train, _ = subset_data(
        train_index, X_cheat, sites, Y
    )

    X_test, sites_test, y_test, covars_test, _ = subset_data(
        test_index, X_cheat, sites, Y)

    pred_model.fit(X_train, y_train)
    pred_cheat, acc_cheat, auc_cheat = \
        compute_metric(pred_model, X_test, y_test,
                       pred_cheat, acc_cheat, auc_cheat)

    pred_cheat_trn, acc_cheat_trn, auc_cheat_trn = \
        compute_metric(pred_model, X_train, y_train,
                       pred_cheat_trn, acc_cheat_trn, auc_cheat_trn,
                       )

# %% Saving
save_dir = "/home/nnieto/Nico/Harmonization/data/eICU/Results/10_images_min/"

# Create the list of lists and save names
save_list = [[save_dir+"eICU_pred_none", pred_none],
             [save_dir+"eICU_pred_target", pred_target],
             [save_dir+"eICU_pred_notarget", pred_notarget],
             [save_dir+"eICU_pred_pretend", pred_pretend],
             [save_dir+"eICU_pred_cheat", pred_cheat],
             [save_dir+"eICU_y_test_list", y_test_list],
             [save_dir+"eICU_sites_test_list", sites_test_list],
             [save_dir+"eICU_fold_plot", folds_lits],
             [save_dir+"eICU_pred_none", pred_none],
             [save_dir+"eICU_pred_target", pred_target]]

# Save the data objects to separate files using pickle
for save_name, data_save in save_list:
    with open(save_name + '.pickle', 'wb') as file:
        pickle.dump(data_save, file)

# %%
pd.DataFrame(data).to_csv(save_dir + "eICU_data_used.csv")
pd.DataFrame(X_cheat).to_csv(save_dir + "eICU_data_harmonized_cheat_used.csv")

# %%
