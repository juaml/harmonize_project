# %%
import numpy as np
import pandas as pd
from juharmonize import JuHarmonizeClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score,
    roc_auc_score,
)
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

min_images_per_site = 50
# Remove sites with less than a thd number of patients
site_counts = data["site"].value_counts()

# filter the site_ids with less than a thd
mask = site_counts[site_counts > min_images_per_site].index.tolist()

# Filter the sites with the minimun number of patietes
data = data[data['site'].isin(mask)]

# Separate the DataFrame into 'Alive' and 'Expired'
alive_df = data[data['endpoint'] == 'Alive']
expired_df = data[data['endpoint'] == 'Expired']

expired_final = pd.DataFrame()
alive_final = pd.DataFrame()

flag = "Expire"
sites_list = expired_df['site'].unique()
for sites in sites_list:

    if flag =="Expire":
        # 1. Identify the site with the most 'Expired' patients and remove all 'Alive' patients from that site
        site_with_most_expired = expired_df['site'].value_counts().idxmax()
        # Keep the expired patients
        expired_final = pd.concat([expired_final, expired_df[expired_df['site'] == site_with_most_expired]])
        expired_df = expired_df[expired_df['site'] != site_with_most_expired]
        
        alive_final = pd.concat([alive_final, alive_df[alive_df['site'] == site_with_most_expired].sample(n=1, random_state=42)])

        # Remove that site from the alive.
        alive_df = alive_df[alive_df['site'] != site_with_most_expired]
        flag = "Alive"
        continue
    if flag =="Alive":
        # 1. Identify the site with the most 'Expired' patients and remove all 'Alive' patients from that site
        site_with_less_expired = expired_df['site'].value_counts().idxmin()
        # Keep the expired patients
        alive_final = pd.concat([alive_final, alive_df[alive_df['site'] == site_with_less_expired]])
        expired_final = pd.concat([expired_final, expired_df[expired_df['site'] == site_with_less_expired].sample(n=1, random_state=42)])

        # Remove that site from the alive.
        alive_df = alive_df[alive_df['site'] != site_with_less_expired]

        expired_df = expired_df[expired_df['site'] != site_with_less_expired]
        flag = "Expire"
        continue

# Combine the remaining 'Alive' and 'Expired' patients into a new balanced DataFrame
balanced_df = pd.concat([alive_final, expired_final])

# Optionally, shuffle the combined DataFrame to mix the 'Alive' and 'Expired' entries
if not balanced_df.empty:
    balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)



# Calculate the count of each target value (Alive, Expired) for each site
site_target_counts = balanced_df.groupby(['site', 'endpoint']).size().unstack(fill_value=0)

# Calculate the proportion for each target within each site
site_target_proportions = site_target_counts.div(site_target_counts.sum(axis=1), axis=0)

# Combine counts and proportions into a single DataFrame
site_summary = pd.concat([site_target_counts, site_target_proportions], axis=1, keys=['Count', 'Proportion'])

# Display the combined DataFrame
print(site_summary)
# %%
#
X = balanced_df.loc[:, ABG_of_interes].to_numpy()
sites = balanced_df["site"].reset_index()
Y = balanced_df["endpoint"].replace({"Expired": 1, "Alive": 0}).to_numpy()


# %%
results = []

kf_out = RepeatedStratifiedKFold(n_splits=5,
                                 n_repeats=1,
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


pred_juharmonize = []
pred_cheat = []
pred_leakage = []
pred_none = []
pred_notarget = []
pred_y_true_loop = []


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
    pred_y_true_loop.append(Y_test)
    # None model
    clf.fit(X_train, Y_train)
    pred_test = clf.predict_proba(X_test)[:, 1]
    results = compute_results(i_fold, "None Test", pred_test, Y_test, results)                 # noqa
    pred_none.append(pred_test)
    pred_train = clf.predict_proba(X_train)[:, 1]
    results = compute_results(i_fold, "None Train", pred_train, Y_train, results)                 # noqa

    # Cheat
    clf.fit(X_cheat_train, Y_train)
    pred_test = clf.predict_proba(X_cheat_test)[:, 1]
    results = compute_results(i_fold, "Cheat Test", pred_test, Y_test, results)                 # noqa
    pred_cheat.append(pred_test)

    pred_train = clf.predict_proba(X_cheat_train)[:, 1]
    results = compute_results(i_fold, "Cheat Train", pred_train, Y_train, results)                 # noqa

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
    pred_leakage.append(pred_test)

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
    pred_notarget.append(pred_test)

    pred_train = clf.predict_proba(harm_data)[:, 1]
    results = compute_results(i_fold, "No Target Train", pred_train, Y_train, results)                 # noqa

    # # JuHarmonize
    JuHarmonize_model.fit(X=X_train, y=Y_train,
                          sites=site_train["site"].to_numpy())
    pred_test = JuHarmonize_model.predict_proba(X_test,
                                                sites=site_test["site"].to_numpy())[:, 1]           # noqa
    results = compute_results(i_fold, "JuHarmonize Test", pred_test, Y_test, results)                 # noqa
    pred_juharmonize.append(pred_test)

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


# results = results_to_df(results)
# save_dir = "/home/nnieto/Nico/Harmonization/harmonize_project/scratch/output/results/sepsis_classification_eicu/"
# results.to_csv(save_dir+"eiCU_results_dependance.csv")
# %%

# %%
# save_dir = "/home/nnieto/Nico/Harmonization/harmonize_project/scratch/output/results/sepsis_classification_eicu/predictions_dependance/"


# pd.DataFrame(pred_juharmonize).to_csv(save_dir + "pred_juharmonize.csv")
# pd.DataFrame(pred_cheat).to_csv(save_dir + "pred_cheat.csv")
# pd.DataFrame(pred_leakage).to_csv(save_dir + "pred_leakage.csv")
# pd.DataFrame(pred_none).to_csv(save_dir + "pred_none.csv")
# pd.DataFrame(pred_notarget).to_csv(save_dir + "pred_notarget.csv")
# pd.DataFrame(pred_y_true_loop).to_csv(save_dir + "y_true_loop.csv")

# %%

import seaborn as sbn

pred_juharmonize = pd.DataFrame(pred_juharmonize)
pred_cheat = pd.DataFrame(pred_cheat)
pred_y_true_loop = pd.DataFrame(pred_y_true_loop)
pred_cheat = pd.DataFrame(pred_none)

# %%

pred_juharmonize_ravel = pred_juharmonize.to_numpy().ravel()
pred_cheat_ravel = pred_cheat.to_numpy().ravel()
pred_true_ravel = pred_y_true_loop.to_numpy().ravel()


# %%

sbn.swarmplot(y = pred_juharmonize_ravel-pred_cheat_ravel)
# %%

prediction_juha = (pred_juharmonize_ravel > 0.5).astype(int)
prediction_chat = (pred_cheat_ravel > 0.5).astype(int)

combined = prediction_juha + 2*prediction_chat
# %%
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 1, figsize=[20, 10])

combined = pd.DataFrame(combined)
combined.replace({0:"Both agree Alive",
1:"JuHarmonize Expire - Cheat  alive",
2:"JuHarmonize Alive - Cheat  Expired",
3:"Both agree expired"}, inplace=True)
combined = combined.to_numpy().squeeze()
sbn.swarmplot(x = combined, y = pred_juharmonize_ravel-pred_cheat_ravel, hue=pred_true_ravel, ax=ax)
plt.ylabel("Prediction difference (JuHarmonize - Cheat)")
plt.legend(["Alive", "Expired"])
plt.grid()
# %%
fig, ax = plt.subplots(1, 1, figsize=[10, 7])
markers = pd.DataFrame(combined).copy()
markers.replace({0:"c",
1:"X",
2:"X",
3:"c"}, inplace=True)

markers={"Both agree Alive":"X",
         "Expired": "X",
         }

combined = pd.DataFrame(combined)
combined.replace({0:"Both agree Alive",
1:"JuHarmonize Expire - None  alive",
2:"JuHarmonize Alive - Cheat  Expired",
3:"Both agree expired"}, inplace=True)
combined = combined.to_numpy().squeeze()




sbn.scatterplot(x=pred_juharmonize_ravel,y=pred_cheat_ravel, hue=combined, markers=markers, ax=ax)
plt.plot([0.5,0.5],[0,1], color="black")
plt.plot([0,1],[0.5,0.5], color="black")
plt.xlim([0,1])
plt.ylim([0,1])
plt.grid()
plt.xlabel("JuHarmonize prediction")
plt.ylabel("None prediction")

# %%
fig, ax = plt.subplots(1, 1, figsize=[10, 7])
pred_true_ravel
sbn.scatterplot(x=pred_juharmonize_ravel,y=pred_cheat_ravel, hue=pred_true_ravel,ax=ax,markers =pred_true_ravel,
                )
plt.legend(["Alive","_asd", "Expired"])

plt.plot([0.5,0.5],[0,1], color="black")
plt.plot([0,1],[0.5,0.5], color="black")
plt.xlim([0,1])
plt.ylim([0,1])

plt.grid()
plt.xlabel("JuHarmonize prediction")
plt.ylabel("None prediction")

# %%
combined = pd.DataFrame(combined)
combined.value_counts()

disagreement = 100*(37-20)/ (763+102+27+23)
