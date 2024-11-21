# %%
import pandas as pd
from sklearn.linear_model import LogisticRegression
from prettyharmonize import PrettYharmonizeClassifier
from neuroHarmonize import harmonizationLearn, harmonizationApply
from sklearn.model_selection import RepeatedStratifiedKFold
from lib.data_processing import compute_classification_results


root_dir = "/home/nnieto/Nico/Harmonization/data/final_data_split/"

data_enki = pd.read_csv(root_dir+"X_eNKI_gender_imbalance_extreme.csv")
data_CamCAN = pd.read_csv(root_dir+"X_CamCAN_gender_imbalance_extreme.csv")

y_enki = pd.read_csv(root_dir+"Y_eNKI_gender_imbalance_extreme.csv")
y_enki["site"] = "eNKI"
y_CamCAN = pd.read_csv(root_dir+"Y_CamCAN_gender_imbalance_extreme.csv")
y_CamCAN["site"] = "CamCAN"

# %%

X = pd.concat([data_CamCAN, data_enki])
X.dropna(axis=1, inplace=True)
X = X.to_numpy()
target = pd.concat([y_CamCAN, y_enki])
# %%
target["site"].replace({"eNKI": 0, "CamCAN": 1}, inplace=True)
sites = target["site"].reset_index()
target["gender"].replace({"F": 0, "M": 1}, inplace=True)

Y = target["gender"]

clf = LogisticRegression()
Pretty_harmonize_model = PrettYharmonizeClassifier(stack_model="logit",
                                                   pred_model="logit")

print("Number of sites: " + str(sites.nunique()))
print("Number of classes: " + str(Y.value_counts()))
# %%
results = []

kf_out = RepeatedStratifiedKFold(n_splits=5,
                                 n_repeats=5,
                                 random_state=23)

covars = pd.DataFrame(sites["site"].to_numpy(), columns=['SITE'])

covars['Target'] = Y.to_numpy().ravel()

harm_cheat, data_cheat = harmonizationLearn(data=X, # noqa
                                            covars=covars)
data_cheat = pd.DataFrame(data_cheat)
Y = Y.to_numpy()

# %%
for i_fold, (train_index, test_index) in enumerate(kf_out.split(X=X, y=Y)):       # noqa
    print("FOLD: " + str(i_fold))

    # Patients used for train and internal XGB validation
    X_train = X[train_index, :]
    X_cheat_train = data_cheat.iloc[train_index, :]

    site_train = sites.iloc[train_index, :]
    Y_train = Y[train_index]

    # Patients used to generete a prediction
    X_test = X[test_index, :]
    X_cheat_test = data_cheat.iloc[test_index, :]

    site_test = sites.iloc[test_index, :]

    Y_test = Y[test_index]

    # None model
    clf.fit(X_train, Y_train)
    pred_test = clf.predict_proba(X_test)[:, 1]
    results = compute_classification_results(i_fold, "None Test", pred_test, Y_test, results)                 # noqa

    pred_train = clf.predict_proba(X_train)[:, 1]
    results = compute_classification_results(i_fold, "None Train", pred_train, Y_train, results)                 # noqa

    # Cheat
    clf.fit(X_cheat_train, Y_train)
    pred_test = clf.predict_proba(X_cheat_test)[:, 1]
    results = compute_classification_results(i_fold, "WDH Test", pred_test, Y_test, results)                 # noqa

    pred_train = clf.predict_proba(X_cheat_train)[:, 1]
    results = compute_classification_results(i_fold, "WDH Train", pred_train, Y_train, results)                 # noqa

    # Leakage
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
    results = compute_classification_results(i_fold, "Leakage Test", pred_test, Y_test, results)                 # noqa

    pred_train = clf.predict_proba(harm_data)[:, 1]
    results = compute_classification_results(i_fold, "Leakage Train", pred_train, Y_train, results)                 # noqa

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

    # # Pretty harmonize
    Pretty_harmonize_model.fit(X=X_train, y=Y_train,
                               sites=site_train["site"].to_numpy())
    pred_test = Pretty_harmonize_model.predict_proba(X_test,
                                                sites=site_test["site"].to_numpy())[:, 1]           # noqa
    results = compute_classification_results(i_fold, "JuHarmonize Test", pred_test, Y_test, results)                 # noqa

    pred_train = Pretty_harmonize_model.predict_proba(X_train,
                                                 sites=site_train["site"].to_numpy())[:, 1]         # noqa
    results = compute_classification_results(i_fold, "JuHarmonize Train", pred_train, Y_train, results)                 # noqa


# %%


results = pd.DataFrame(results,
                       columns=["Fold",
                                "Model",
                                "Random State",
                                "Random Permutation Number",
                                "Thresholds",
                                "Number of Removed Features",
                                "Balanced ACC",
                                "AUC",
                                "F1",
                                "Recall",
                                ])



# %%
results.to_csv("/home/nnieto/Nico/Harmonization/harmonize_project/scratch/output/results/sex_classification/results_JuHarmonize.csv")   # noqa
