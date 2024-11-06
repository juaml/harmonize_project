# %%
import pandas as pd
import numpy as np
from juharmonize import JuHarmonizeRegressor
from neuroHarmonize import harmonizationLearn, harmonizationApply
from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import mean_absolute_error, r2_score
from typing import List, Union
from skrvm import RVR

# %%


def compute_results(i_fold: int, model: str,
                    pred: np.ndarray,
                    y: np.ndarray,
                    result: List[List[Union[int, str, float]]],
                    ) -> List[List[Union[int, str, float]]]:
    """
    Calculate evaluation metrics by fold and append results to the given list.
    Parameters:
        i_fold (int): Index of the fold.
        model (str): Model name or identifier.
        prob (np.ndarray): Probability predictions.
        y (np.ndarray): True labels.

        result (List[List[Union[int, str, float]]]): List to store the results.

    Returns:
        List[List[Union[int, str, float]]]: Updated list with appended results.
    """

    # Calculate the predictions using the passed ths

    # Compute all the metrics
    mae = mean_absolute_error(y, pred)
    r2 = r2_score(y, pred)
    age_bias = np.corrcoef(y, pred-y)[0, 1]

    # Append results
    result.append([i_fold, model, mae, r2, age_bias])

    return result


def balance_gender(data, min_images=59):
    male = data[data["gender"] == "M"]
    female = data[data["gender"] == "F"]

    male = male.sample(n=min_images, random_state=23)
    female = female.sample(n=min_images, random_state=23)

    data_balanced = pd.concat([male, female])

    return data_balanced


def load_crop_dataset(name, data_dir):
    Y = pd.read_csv(data_dir+"Y_"+name+"_crop.csv")
    X = pd.read_csv(data_dir+"X_"+name+"_crop.csv")
    return X, Y


def retain_images(X, Y):
    return X.loc[Y.index, :]
# %%


data_dir = "/home/nnieto/Nico/Harmonization/data/final_data_split/"

X_ID1000, Y_ID1000 = load_crop_dataset("ID1000", data_dir)
X_eNKI, Y_eNKI = load_crop_dataset("eNKI", data_dir)
X_Camcan, Y_Camcan = load_crop_dataset("CamCAN", data_dir)
X_1000brains, Y_1000brains = load_crop_dataset("1000Gehirne", data_dir)

Y_ID1000 = balance_gender(Y_ID1000)
Y_eNKI = balance_gender(Y_eNKI)
Y_Camcan = balance_gender(Y_Camcan)
Y_1000brains = balance_gender(Y_1000brains)

Y = pd.concat([Y_ID1000, Y_eNKI, Y_Camcan, Y_1000brains])

X = pd.concat([retain_images(X_ID1000, Y_ID1000),
               retain_images(X_eNKI, Y_eNKI),
               retain_images(X_Camcan, Y_Camcan),
               retain_images(X_1000brains, Y_1000brains)])

X.dropna(axis=1, inplace=True)
# %%

Y["site"].replace({"ID1000": 0, "eNKI": 1,
                   "CamCAN": 2, "1000Gehirne": 3}, inplace=True)
sites = Y["site"].reset_index()
Y["gender"].replace({"F": 0, "M": 1}, inplace=True)

Y = Y["age"].to_numpy()
X = X.to_numpy()

# %%
results = []

kf_out = RepeatedKFold(n_splits=5,
                       n_repeats=1,
                       random_state=23)

covars = pd.DataFrame(sites["site"].to_numpy(), columns=['SITE'])

covars['Target'] = Y.ravel()

harm_cheat, data_cheat = harmonizationLearn(data=X, # noqa
                                            covars=covars)
data_cheat = pd.DataFrame(data_cheat)


# %%
stack_model = RVR(kernel="poly", degree=1)
pred_model = RVR(kernel="poly", degree=1)

JuHarmonize_model = JuHarmonizeRegressor(stack_model=stack_model,
                                         pred_model=pred_model)


y_true_loop = []
sites_loop = []
pred_none = []
pred_cheat = []
pred_notarget = []
pred_leak = []
pred_juharmonize = []

save_dir = "/home/nnieto/Nico/Harmonization/harmonize_project/scratch/output/"
# %%
for i_fold, (train_index, test_index) in enumerate(kf_out.split(X=X)):       # noqa
    print("FOLD: " + str(i_fold))
    # Train
    X_train = X[train_index, :]
    X_cheat_train = data_cheat.iloc[train_index, :]

    site_train = sites.iloc[train_index, :]
    Y_train = Y[train_index]

    # Test
    X_test = X[test_index, :]
    X_cheat_test = data_cheat.iloc[test_index, :]

    site_test = sites.iloc[test_index, :]
    sites_loop.append(site_test["site"].to_numpy())
    Y_test = Y[test_index]
    y_true_loop.append(Y_test)

    # Models
    # None model
    clf = RVR(kernel="poly", degree=1)

    clf.fit(X_train, Y_train)
    pred_test = clf.predict(X_test)
    results = compute_results(i_fold, "None Test", pred_test, Y_test, results)                 # noqa
    pred_none.append(pred_test)

    # Cheat
    clf = RVR(kernel="poly", degree=1)

    clf.fit(X_cheat_train, Y_train)
    pred_test = clf.predict(X_cheat_test)
    results = compute_results(i_fold, "Cheat Test", pred_test, Y_test, results)                 # noqa
    pred_cheat.append(pred_test)

    # # Leakage
    clf = RVR(kernel="poly", degree=1)

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

    pred_test = clf.predict(harm_data_test)
    results = compute_results(i_fold, "Leakage Test", pred_test, Y_test, results)                       # noqa
    pred_leak.append(pred_test)

    # No Target
    clf = RVR(kernel="poly", degree=1)

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

    pred_test = clf.predict(harm_data_test)
    results = compute_results(i_fold, "No Target Test", pred_test, Y_test, results)                     # noqa
    pred_notarget.append(pred_test)

    # # JuHarmonize
    JuHarmonize_model.fit(X=X_train, y=Y_train,
                          sites=site_train["site"].to_numpy())
    pred_test = JuHarmonize_model.predict(X_test,
                                        sites=site_test["site"].to_numpy())                     # noqa
    results = compute_results(i_fold, "JuHarmonize Test", pred_test, Y_test, results)                   # noqa
    pred_juharmonize.append(pred_test)

# %%
pd.DataFrame(y_true_loop).to_csv(save_dir+"y_true.csv")
pd.DataFrame(sites_loop).to_csv(save_dir+"sites.csv")

pd.DataFrame(pred_none).to_csv(save_dir+"y_pred_none.csv")
pd.DataFrame(pred_cheat).to_csv(save_dir+"y_pred_cheat.csv")
pd.DataFrame(pred_notarget).to_csv(save_dir+"y_pred_notarget.csv")
pd.DataFrame(pred_leak).to_csv(save_dir+"y_pred_leak.csv")
pd.DataFrame(pred_juharmonize).to_csv(save_dir+"y_pred_juharmonize.csv")

pd.DataFrame(results, columns=["Fold", "Harmonization Scheme", "MAE", "R2",
                               "Age bias"]
             ).to_csv(save_dir+"results/results_age_regression_balanced_disjoint_ranges.csv")       # noqa
# %%
sites_print = [Y_ID1000, Y_eNKI, Y_Camcan, Y_1000brains]

for site in sites_print:
    print(site["age"].mean())
    print(site["age"].std())
    print(site["age"].min())
    print(site["age"].max())
