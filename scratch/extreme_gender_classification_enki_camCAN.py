# %%
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from juharmonize import JuHarmonizeClassifier
from sklearn.metrics import balanced_accuracy_score
from neuroHarmonize import harmonizationLearn, harmonizationApply
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import f1_score, recall_score
from sklearn.metrics import roc_auc_score
from typing import List, Union

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
JuHarmonize_model = JuHarmonizeClassifier(stack_model="logit",
                                          pred_model="logit")

print("Number of sites: " + str(sites.nunique()))
print("Number of classes: " + str(Y.value_counts()))
# %%
results = []

kf_out = RepeatedStratifiedKFold(n_splits=5,
                                 n_repeats=5,
                                 random_state=23)

covars = pd.DataFrame(sites["site"].to_numpy(), columns=['SITE'])


harm_cheat, data_cheat_no_target = harmonizationLearn(data=X, # noqa
                                                      covars=covars)

covars['Target'] = Y.to_numpy().ravel()

harm_cheat, data_cheat = harmonizationLearn(data=X, # noqa
                                            covars=covars)
data_cheat = pd.DataFrame(data_cheat)
Y = Y.to_numpy()


def compute_results(i_fold: int, model: str,
                    prob: np.ndarray,
                    y: np.ndarray,
                    result: List[List[Union[int, str, float]]],
                    rs: bool = False,
                    rpn: int = 0,
                    ths_range: Union[float, List[float]] = 0.5,
                    n_removed_features: int = 0,
                    ) -> List[List[Union[int, str, float]]]:
    """
    Calculate evaluation metrics by fold and append results to the given list.
    # noqa
    Parameters:
        i_fold (int): Index of the fold.
        model (str): Model name or identifier.
        rs (bool): Random State.
        rpn (int): Random Permutation number.
        prob (np.ndarray): Probability predictions.
        y (np.ndarray): True labels.
        ths_range (Union[float, List[float]]): Thresholds for binary classification.
        n_removed_features (int): Number of removed features.
        result (List[List[Union[int, str, float]]]): List to store the results.

    Returns:
        List[List[Union[int, str, float]]]: Updated list with appended results.
    """
    # If a float value was provided, convert in list for iteration
    if isinstance(ths_range, float):
        ths_range = [ths_range]

    for ths in ths_range:

        # Calculate the predictions using the passed ths
        prediction = (prob > ths).astype(int)

        # Compute all the metrics
        bacc = balanced_accuracy_score(y, prediction)
        auc = roc_auc_score(y, prob)
        f1 = f1_score(y, prediction)
        recall = recall_score(y, prediction)

        # Append results
        result.append([i_fold, model, rs, rpn, ths, n_removed_features,
                       bacc, auc, f1,
                       recall,
                       ])

    return result

# %%
for i_fold, (train_index, test_index) in enumerate(kf_out.split(X=X, y=Y)):       # noqa
    print("FOLD: " + str(i_fold))

    # Patients used for train and internal XGB validation
    X_train = X[train_index, :]
    X_cheat_train = data_cheat.iloc[train_index, :]
    X_cheat_no_target_train = data_cheat_no_target[train_index, :]

    site_train = sites.iloc[train_index, :]
    Y_train = Y[train_index]

    # Patients used to generete a prediction
    X_test = X[test_index, :]
    X_cheat_test = data_cheat.iloc[test_index, :]
    X_cheat_no_target_test = data_cheat_no_target[test_index, :]

    site_test = sites.iloc[test_index, :]

    Y_test = Y[test_index]

    # None model
    clf.fit(X_train, Y_train)
    pred_test = clf.predict_proba(X_test)[:, 1]
    results = compute_results(i_fold, "None Test", pred_test, Y_test, results)                 # noqa

    pred_train = clf.predict_proba(X_train)[:, 1]
    results = compute_results(i_fold, "None Train", pred_train, Y_train, results)                 # noqa

    # Cheat
    clf.fit(X_cheat_train, Y_train)
    pred_test = clf.predict_proba(X_cheat_test)[:, 1]
    results = compute_results(i_fold, "Cheat Test", pred_test, Y_test, results)                 # noqa

    pred_train = clf.predict_proba(X_cheat_train)[:, 1]
    results = compute_results(i_fold, "Cheat Train", pred_train, Y_train, results)                 # noqa

    # Cheat no target
    clf.fit(X_cheat_no_target_train, Y_train)
    pred_test = clf.predict_proba(X_cheat_no_target_test)[:, 1]
    results = compute_results(i_fold, "Cheat No Target Test", pred_test, Y_test, results)                 # noqa

    pred_train = clf.predict_proba(X_cheat_no_target_train)[:, 1]
    results = compute_results(i_fold, "Cheat No Target Train", pred_train, Y_train, results)                 # noqa


    # # Leakage
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
    results = compute_results(i_fold, "Leakage Test", pred_test, Y_test, results)                 # noqa

    pred_train = clf.predict_proba(harm_data)[:, 1]
    results = compute_results(i_fold, "Leakage Train", pred_train, Y_train, results)                 # noqa

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
    results = compute_results(i_fold, "No Target Test", pred_test, Y_test, results)                 # noqa

    pred_train = clf.predict_proba(harm_data)[:, 1]
    results = compute_results(i_fold, "No Target Train", pred_train, Y_train, results)                 # noqa

    # # JuHarmonize
    JuHarmonize_model.fit(X=X_train, y=Y_train,
                          sites=site_train["site"].to_numpy())
    pred_test = JuHarmonize_model.predict_proba(X_test,
                                                sites=site_test["site"].to_numpy())[:, 1]           # noqa
    results = compute_results(i_fold, "JuHarmonize Test", pred_test, Y_test, results)                 # noqa

    pred_train = JuHarmonize_model.predict_proba(X_train,
                                                 sites=site_train["site"].to_numpy())[:, 1]         # noqa
    results = compute_results(i_fold, "JuHarmonize Train", pred_train, Y_train, results)                 # noqa


# %%
def results_to_df(result: List[List[Union[int, str, float]]]) -> pd.DataFrame:
    """
    Convert the list of results to a DataFrame.

    Parameters:
        result (List[List[Union[int, str, float]]]): List containing results.

    Returns:
        pd.DataFrame: DataFrame containing results with labeled columns.
    """
    result_df = pd.DataFrame(result,
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
    return result_df


results = results_to_df(results)
# %%
results.to_csv("/home/nnieto/Nico/Harmonization/harmonize_project/scratch/output/results/sex_classification/results_JuHarmonize.csv")   # noqa
# %%
import seaborn as sbn
data = results
# site_order = ["Global", "eNKI", "CamCAN"]
metric_to_plot = "Balanced ACC"
import matplotlib.pyplot as plt
# Plot
pal = sbn.cubehelix_palette(5, rot=-.5, light=0.5, dark=0.2)
_, ax = plt.subplots(1, 1, figsize=[12, 7])


sbn.boxplot(
    data=data, zorder=1,
    x="Model", y=metric_to_plot, hue="Model",
 dodge=False, ax=ax
)

plt.ylabel(metric_to_plot)
plt.xlabel("Sites")
plt.title("Gender Classification")
plt.grid(alpha=0.5, axis="y", c="black")
plt.show()