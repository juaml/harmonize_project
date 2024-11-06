# %%
import pandas as pd
import numpy as np

from juharmonize import JuHarmonizeClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score,
    roc_auc_score,
)


from neuroHarmonize import harmonizationLearn, harmonizationApply
from sklearn.metrics import f1_score, recall_score
from typing import List, Union
data_dir = "/home/nnieto/Nico/Harmonization/data/ADNI/Kersten_data/final_data_split/"       # noqa
X1 = pd.read_csv(data_dir+"X_SITE_1.csv")
X2 = pd.read_csv(data_dir+"X_SITE_2.csv")
Y1 = pd.read_csv(data_dir+"Y_SITE_1.csv")
Y2 = pd.read_csv(data_dir+"Y_SITE_2.csv")

# Assuming you have the following DataFrames: X1, Y1, X2, Y2
random_state = 23

# Generate dependance
# Select 100 "F" and 1 "M" from Y1
Y1_females = Y1[Y1['gender'] == 'F'].sample(n=10, random_state=random_state)
Y1_male = Y1[Y1['gender'] == 'M'].sample(n=100, random_state=random_state)

# Combine the selected "F" and "M" to form the new Y1
new_Y1 = pd.concat([Y1_females, Y1_male])

# Ensure X1 is synchronized with the new Y1 by selecting the corresponding rows
new_X1 = X1.loc[new_Y1.index]

# Select 100 "M" and 1 "F" from Y2
Y2_males = Y2[Y2['gender'] == 'M'].sample(n=10, random_state=random_state)
Y2_female = Y2[Y2['gender'] == 'F'].sample(n=100, random_state=random_state)

# Combine the selected "M" and "F" to form the new Y2
new_Y2 = pd.concat([Y2_males, Y2_female])

# Ensure X2 is synchronized with the new Y2 by selecting the corresponding rows
new_X2 = X2.loc[new_Y2.index]

# Display the new DataFrames
print("new_Y1:")
print(new_Y1)
print("new_X1:")
print(new_X1)
print("new_Y2:")
print(new_Y2)
print("new_X2:")
print(new_X2)

# %%

X = pd.concat([new_X1, new_X2])
Y = pd.concat([new_Y1, new_Y2])

# %%

X = X.to_numpy()
sites = Y["site"].reset_index()
Y = Y["gender"].replace({"F": 1, "M": 0}).to_numpy()


# %%
results = []

kf_out = RepeatedStratifiedKFold(n_splits=5,
                                 n_repeats=5,
                                 random_state=23)


covars = pd.DataFrame(sites["site"].to_numpy(), columns=['SITE'])


harm_cheat, data_cheat_no_target = harmonizationLearn(data=X, # noqa
                                                      covars=covars)

covars['Target'] = Y.ravel()

harm_cheat, data_cheat = harmonizationLearn(data=X, # noqa
                                            covars=covars)
data_cheat = pd.DataFrame(data_cheat)
clf = LogisticRegression()
JuHarmonize_model = JuHarmonizeClassifier(stack_model="logit",
                                          pred_model="logit")


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
save_dir = "/home/nnieto/Nico/Harmonization/harmonize_project/scratch/output/results/Kersten/"  # noqa
results.to_csv(save_dir+"Kersten_results_dependance.csv")
# %%
