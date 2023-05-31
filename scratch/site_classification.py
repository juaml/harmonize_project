# %%
import pandas as pd
import seaborn as sbn
import matplotlib.pyplot as plt
import numpy as np
from juharmonize import JuHarmonize
from juharmonize.utils import subset_data
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import RidgeClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    balanced_accuracy_score,
    roc_auc_score,
    accuracy_score,
    mean_absolute_error,
    r2_score
)
from sklearn.model_selection import GridSearchCV


def compute_metric(model, X, y, predicted_y, acc, auc_score):
    predicted_y_loop = model.predict(X)
    # predicted_y_loop_proba = model.predict_proba(X)
    acc_loop = 0
    auc_score_loop = balanced_accuracy_score(y, predicted_y_loop)
    predicted_y.append(predicted_y_loop)
    acc = np.append(acc, acc_loop)
    auc_score = np.append(auc_score, auc_score_loop)
    return predicted_y, acc, auc_score


Y_1000brains = pd.read_csv("/home/nnieto/Nico/Harmonization/data/final_data_split/Y_1000Gehirne_crop.csv")  # noqa
Y_Camcan = pd.read_csv("/home/nnieto/Nico/Harmonization/data/final_data_split/Y_CamCAN_crop.csv")           # noqa
Y_ID1000 = pd.read_csv("/home/nnieto/Nico/Harmonization/data/final_data_split/Y_ID1000.csv")                # noqa
Y_eNKI = pd.read_csv("/home/nnieto/Nico/Harmonization/data/final_data_split/Y_eNKI_crop.csv")               # noqa

data = pd.concat([Y_1000brains, Y_Camcan, Y_eNKI, Y_ID1000])

X_1000brains = pd.read_csv("/home/nnieto/Nico/Harmonization/data/final_data_split/X_1000Gehirne_crop.csv", index_col=0)  # noqa
X_Camcan = pd.read_csv("/home/nnieto/Nico/Harmonization/data/final_data_split/X_CamCAN_crop.csv", index_col=0)           # noqa
X_ID1000 = pd.read_csv("/home/nnieto/Nico/Harmonization/data/final_data_split/X_ID1000.csv", index_col=0)                # noqa
X_eNKI = pd.read_csv("/home/nnieto/Nico/Harmonization/data/final_data_split/X_eNKI_crop.csv", index_col=0)               # noqa

X_total = pd.concat([X_1000brains, X_Camcan, X_eNKI, X_ID1000])
X_total.dropna(axis=1, inplace=True)
X = X_total.to_numpy()
# data["site"].replace({"ID1000": 0,
#                       "1000Gehirne": 1,
#                       "eNKI": 2,
#                       "CamCAN": 3},
#                      inplace=True)

del X_1000brains, X_Camcan, X_eNKI, X_ID1000
del Y_1000brains, Y_Camcan, Y_eNKI, Y_ID1000
# %%
N_images = 105

# Create a list of unique site names in the data dataframe
sites = data['site'].unique()

# Create an empty dataframe to store the filtered data
filtered_data = pd.DataFrame()
filtered_data_X = pd.DataFrame()

# For each site, randomly select N_images and append them
# to the filtered_data dataframes
for site in sites:
    # Filter the data dataframe for the current site
    site_data = data[data['site'] == site]

    # Randomly select N_images from the current site's data dataframe
    selected_indices = np.random.choice(len(site_data), N_images,
                                        replace=False)
    selected_data = site_data.iloc[selected_indices]

    # Filter the data_X dataframe for the selected rows of data
    selected_data_X = X_total.iloc[selected_data.index]

    # Append the selected data and data_X to the filtered dataframes
    filtered_data = filtered_data.append(selected_data)
    filtered_data_X = filtered_data_X.append(selected_data_X)

data = filtered_data
X_total = filtered_data_X

X = X_total.to_numpy()
sites = data["site"].to_numpy()
Y = data["age"].to_numpy()

scaler = RobustScaler()
models_pars = {"alphas": [0.01, 0.1, 1, 10, 100]}
model = RidgeClassifierCV()

# models_pars = {"n_estimators": [5, 10, 20, 40, 60, 80, 100]}
# model = RandomForestClassifier()


pred_model = GridSearchCV(model, param_grid=models_pars, n_jobs=-1)
cheat_model = JuHarmonize()
# X = scaler.fit_transform(X, Y)
X_cheat = cheat_model.fit_transform(X, Y, sites)
kf = RepeatedStratifiedKFold(n_splits=3, n_repeats=1,
                             random_state=43)
print(data["site"].value_counts())
# %%
# %%

pred_cheat = []
pred_none = []

pred_cheat_trn = []
pred_none_trn = []

acc_none = []
acc_cheat = []

auc_none = []
auc_cheat = []

acc_none_trn = []
acc_cheat_trn = []

auc_none_trn = []
auc_cheat_trn = []

y_test_list = []
y_train_list = []
sites_train_list = []
sites_test_list = []
folds_lits = []

for i_fold, (train_index, test_index) in enumerate(kf.split(X, sites)):

    print("FOLD: " + str(i_fold))
    X_train, sites_train, y_train, covars_train, _ = subset_data(
        train_index, X, sites, Y
    )

    X_test, sites_test, y_test, covars_test, _ = subset_data(
        test_index, X, sites, Y)

    sites_train_list.append(sites_train)
    sites_test_list.append(sites_test)

    # None
    pred_model.fit(X_train, sites_train)
    pred_none, acc_none, auc_none = \
        compute_metric(pred_model, X_test, sites_test,
                       pred_none, acc_none, auc_none)

    pred_none_trn, acc_none_trn, auc_none_trn = \
        compute_metric(pred_model, X_train, sites_train,
                       pred_none_trn, acc_none_trn, auc_none_trn)

    # Cheat
    X_train_cheat, sites_train, y_train, covars_train, _ = subset_data(
        train_index, X_cheat, sites, Y)

    X_test_test, sites_test, y_test, covars_test, _ = subset_data(
        test_index, X_cheat, sites, Y)

    pred_model.fit(X_train_cheat, sites_train)
    pred_cheat, acc_cheat, auc_cheat = \
        compute_metric(pred_model, X_test_test, sites_test,
                       pred_cheat, acc_cheat, auc_cheat)

    pred_cheat_trn, acc_cheat_trn, auc_cheat_trn = \
        compute_metric(pred_model, X_train_cheat, sites_train,
                       pred_cheat_trn, acc_cheat_trn, auc_cheat_trn,
                       )

# %%

print(auc_cheat.mean())
print(auc_none.mean())

# %%
print(auc_cheat_trn.mean())
print(auc_none_trn.mean())
# %%
