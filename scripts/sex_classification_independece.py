# %%
import pandas as pd
from sklearn.linear_model import LogisticRegression
from prettyharmonize import PrettYharmonizeClassifier
from neuroHarmonize import harmonizationLearn, harmonizationApply
from sklearn.model_selection import RepeatedStratifiedKFold
import sys
import os
project_root = os.path.dirname((os.path.dirname(os.path.abspath(__file__))))             # noqa
sys.path.append(project_root)

from lib.data_processing import compute_classification_results                          # noqa
from lib.data_loading import load_sex_age_balanced_data                                 # noqa


# %%
save_dir = "/output/age_regression/"
data_dir = "/balanced/final_data_split/"


X, Y, sites = load_sex_age_balanced_data(data_dir=data_dir)


# %%
clf = LogisticRegression()
PrettYharmonize_model = PrettYharmonizeClassifier(stack_model="logit",
                                                  pred_model="logit")

print("Number of sites: " + str(sites.nunique()))
# %%
results = []

kf_out = RepeatedStratifiedKFold(n_splits=5,
                                 n_repeats=5,
                                 random_state=23)

covars = pd.DataFrame(sites["site"].to_numpy(), columns=['SITE'])


harm_WDH, data_WDH_no_target = harmonizationLearn(data=X, # noqa
                                                  covars=covars)

covars['Target'] = Y.ravel()

harm_WDH, data_WDH = harmonizationLearn(data=X, # noqa
                                        covars=covars)
data_WDH = pd.DataFrame(data_WDH)

# %%
for i_fold, (train_index, test_index) in enumerate(kf_out.split(X=X, y=Y)):       # noqa
    print("FOLD: " + str(i_fold))

    # Patients used for train and internal XGB validation
    X_train = X[train_index, :]
    X_WDH_train = data_WDH.iloc[train_index, :]
    X_WDH_no_target_train = data_WDH_no_target[train_index, :]

    site_train = sites.iloc[train_index, :]
    Y_train = Y[train_index]

    # Patients used to generete a prediction
    X_test = X[test_index, :]
    X_WDH_test = data_WDH.iloc[test_index, :]
    X_WDH_no_target_test = data_WDH_no_target[test_index, :]

    site_test = sites.iloc[test_index, :]

    Y_test = Y[test_index]

    # Unharmonize model
    clf.fit(X_train, Y_train)
    pred_test = clf.predict_proba(X_test)[:, 1]
    results = compute_classification_results(i_fold, "Unharmonize Test", pred_test, Y_test, results)                 # noqa

    pred_train = clf.predict_proba(X_train)[:, 1]
    results = compute_classification_results(i_fold, "Unharmonize Train", pred_train, Y_train, results)                 # noqa

    # WDH
    clf.fit(X_WDH_train, Y_train)
    pred_test = clf.predict_proba(X_WDH_test)[:, 1]
    results = compute_classification_results(i_fold, "WDH Test", pred_test, Y_test, results)                 # noqa

    pred_train = clf.predict_proba(X_WDH_train)[:, 1]
    results = compute_classification_results(i_fold, "WDH Train", pred_train, Y_train, results)                 # noqa

    # # TTL
    covars_train = pd.DataFrame(site_train["site"].to_numpy(),
                                columns=['SITE'])
    covars_train['Target'] = Y_train.ravel()

    harm_model, harm_data = harmonizationLearn(X_train, covars_train)
    # Fit the model with the harmonizezd trian
    clf.fit(harm_data, Y_train)
    # covars
    covars_test = pd.DataFrame(site_test["site"].to_numpy(),
                               columns=['SITE'])
    covars_test['Target'] = Y_test.ravel()

    harm_data_test = harmonizationApply(X_test,
                                        covars_test,
                                        harm_model)

    pred_test = clf.predict_proba(harm_data_test)[:, 1]
    results = compute_classification_results(i_fold, "TTL Test", pred_test, Y_test, results)                 # noqa

    pred_train = clf.predict_proba(harm_data)[:, 1]
    results = compute_classification_results(i_fold, "TTL Train", pred_train, Y_train, results)                 # noqa

    # No Target
    covars_train = pd.DataFrame(site_train["site"].to_numpy(),
                                columns=['SITE'])

    harm_model, harm_data = harmonizationLearn(X_train, covars_train)
    # Fit the model with the harmonizezd trian
    clf.fit(harm_data, Y_train)
    # covars
    covars_test = pd.DataFrame(site_test["site"].to_numpy(),
                               columns=['SITE'])
    harm_data_test = harmonizationApply(X_test,
                                        covars_test,
                                        harm_model)

    pred_test = clf.predict_proba(harm_data_test)[:, 1]
    results = compute_classification_results(i_fold, "No Target Test", pred_test, Y_test, results)                 # noqa

    pred_train = clf.predict_proba(harm_data)[:, 1]
    results = compute_classification_results(i_fold, "No Target Train", pred_train, Y_train, results)                 # noqa

    # # PrettYharmonize
    PrettYharmonize_model.fit(X=X_train, y=Y_train,
                              sites=site_train["site"].to_numpy())
    pred_test = PrettYharmonize_model.predict_proba(X_test,
                                                sites=site_test["site"].to_numpy())[:, 1]           # noqa
    results = compute_classification_results(i_fold, "PrettYharmonize Test", pred_test, Y_test, results)                 # noqa

    pred_train = PrettYharmonize_model.predict_proba(X_train,
                                                 sites=site_train["site"].to_numpy())[:, 1]         # noqa
    results = compute_classification_results(i_fold, "PrettYharmonize Train", pred_train, Y_train, results)                 # noqa


# %%

results = pd.DataFrame(results,
                       columns=["Fold",
                                "Model",
                                "Balanced ACC",
                                "AUC",
                                "F1",
                                "Recall",
                                ])


# %%
results.to_csv(save_dir+"/results/sex_classification/results_sex_classification_independence.csv")   # noqa
