# %%
import warnings
import numpy as np
import pandas as pd
import seaborn as sbn
import matplotlib.pyplot as plt
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegressionCV
from juharmonize import JuHarmonizeClassifier
from sklearn.ensemble import RandomForestClassifier                     # noqa
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier          # noqa
from neuroHarmonize import harmonizationLearn, harmonizationApply
from sklearn.metrics import balanced_accuracy_score
# %%
root_dir = "/home/nnieto/Nico/Harmonization/data/MAREoS/public_datasets/"
save_dir = "/home/nnieto/Nico/Harmonization/results_classification/MAREoS/"

effects = ["true", "eos"]
effect_types = ["simple", "interaction"]
effect_examples = ["1", "2"]

clf_name = "RF"
random_state = 23

if clf_name == "LASSO":
    clf = LogisticRegressionCV(penalty="l1", solver="liblinear", verbose=False,
                               n_jobs=-1, random_state=random_state)
elif clf_name == "SVM":
    clf = SVC(probability=True, random_state=random_state)
elif clf_name == "RF":

    clf = RandomForestClassifier(n_jobs=-1, random_state=random_state)
elif clf_name == "GP":
    clf = GaussianProcessClassifier(n_jobs=-1, random_state=random_state)

JuHarmonize_model = JuHarmonizeClassifier(pred_model=clf,
                                          stack_model="logit",
                                          random_state=random_state)
warnings.filterwarnings(action="ignore", category=ConvergenceWarning)
results = dict()
Juharmonize_results = []
simple_results = []
cheat_results = []
leakage_results = []
notarget_results = []

target_sites = ['site1', 'site8']

for effect in effects:
    for e_types in effect_types:
        for e_example in effect_examples:

            example = effect+"_"+e_types+e_example
            print("Experiment name: " + example)
            data = pd.read_csv(root_dir+example+"_data.csv", index_col=0)
            sites = data["site"]

            site_mask = np.isin(sites, target_sites)
            data = data[site_mask]
            folds = data["folds"]
            sites = data["site"]

            target = pd.read_csv(root_dir+example+"_response.csv", index_col=0)
            target = target[site_mask]

            # Extract site numbers from strings
            site_numbers = sites.str.extract(r'site(\d+)').astype(int)
            # Convert to NumPy array
            site_array = site_numbers.to_numpy().ravel()
            covars = pd.DataFrame(site_array, columns=['SITE'])
            covars['Target'] = target.to_numpy().ravel()
            data_cheat = data.drop(columns=["cov1", "cov2", "site", "folds"])
            harm_cheat, data_cheat = harmonizationLearn(data=data_cheat.to_numpy(), # noqa
                                                        covars=covars)
            data_cheat = pd.DataFrame(data_cheat)
            data_cheat["folds"] = data["folds"].to_numpy()

            for fold in folds.unique():
                print("Fold Number: " + str(fold))
                # Train Data
                X = data[data["folds"] != fold]
                site_train = X["site"]
                X = X.drop(columns=["cov1", "cov2", "site", "folds"])

                # Load "cheat" data
                X_cheat = data_cheat[data_cheat["folds"] != fold]
                X_cheat = X_cheat.drop(columns=["folds"])
                # Train Target
                y = target[data["folds"] != fold].to_numpy().ravel()
                # Test data
                X_test = data[data["folds"] == fold]
                site_test = X_test["site"]
                X_test = X_test.drop(columns=["cov1", "cov2", "site", "folds"])

                # Get "cheat" data
                X_cheat_test = data_cheat[data_cheat["folds"] == fold]
                X_cheat_test = X_cheat_test.drop(columns=["folds"])
                # Test target
                y_test = target[data["folds"] == fold].to_numpy().ravel()

                # None model
                clf.fit(X, y)
                simple_results.append([balanced_accuracy_score(y_true=y_test,
                                                               y_pred=clf.predict(X=X_test)), fold, effect, e_types, e_example, example])    # noqa

                # Cheat
                clf.fit(X_cheat, y)
                cheat_results.append([balanced_accuracy_score(y_true=y_test,
                                                               y_pred=clf.predict(X=X_cheat_test)),           # noqa  
                                                               fold, effect, e_types, e_example, example])    # noqa

                # Leakage
                # Extract site numbers from strings
                site_numbers = site_train.str.extract(r'site(\d+)').astype(int)
                # Convert to NumPy array
                site_array = site_numbers.to_numpy()
                covars_train = pd.DataFrame(site_array.astype(int),
                                            columns=['SITE'])
                covars_train['Target'] = y
                harm_model, harm_data = harmonizationLearn(X.to_numpy(),
                                                           covars_train)
                # Fit the model with the harmonizezd trian
                clf.fit(harm_data, y)
                # covars
                # Extract site numbers from strings
                site_numbers = site_test.str.extract(r'site(\d+)').astype(int)
                # Convert to NumPy array
                site_array = site_numbers.to_numpy()
                covars_test = pd.DataFrame(site_array.astype(int),
                                           columns=['SITE'])
                covars_test['Target'] = y_test
                harm_data_test = harmonizationApply(X_test.to_numpy(),
                                                    covars_test,
                                                    harm_model)

                leakage_results.append([balanced_accuracy_score(y_true=y_test,
                                                               y_pred=clf.predict(X=harm_data_test)), fold, effect, e_types, e_example, example])    # noqa

                # No Target
                # Extract site numbers from strings
                site_numbers = site_train.str.extract(r'site(\d+)').astype(int)
                # Convert to NumPy array
                site_array = site_numbers.to_numpy()
                covars_train = pd.DataFrame(site_array.astype(int),
                                            columns=['SITE'])
                harm_model, harm_data = harmonizationLearn(X.to_numpy(),
                                                           covars_train)
                # Fit the model with the harmonizezd trian
                clf.fit(harm_data, y)
                # covars
                # Extract site numbers from strings
                site_numbers = site_test.str.extract(r'site(\d+)').astype(int)
                # Convert to NumPy array
                site_array = site_numbers.to_numpy()
                covars_test = pd.DataFrame(site_array.astype(int),
                                           columns=['SITE'])
                harm_data_test = harmonizationApply(X_test.to_numpy(),
                                                    covars_test,
                                                    harm_model)

                notarget_results.append([balanced_accuracy_score(y_true=y_test,
                                                               y_pred=clf.predict(X=harm_data_test)), fold, effect, e_types, e_example, example])    # noqa

                # JuHarmonize
                JuHarmonize_model.fit(X.to_numpy(), y,
                                      sites=site_train.to_numpy())
                y_pred = JuHarmonize_model.predict(X_test.to_numpy(),
                                                   sites=site_test)
                Juharmonize_results.append([balanced_accuracy_score(y_true=y_test,                                                                          # noqa
                                                             y_pred=y_pred), fold, effect, e_types, e_example, example])                                    # noqa


# Save results
results_none = pd.DataFrame(data=simple_results,
                            columns=['bACC', "Fold", "Effect",
                                     "Type", "Example", "Name"])
results_none["Method"] = "None"

results_Juharmonize = pd.DataFrame(data=Juharmonize_results,
                                   columns=['bACC', "Fold", "Effect",
                                            "Type", "Example", "Name"])
results_Juharmonize["Method"] = "JuHarmonize"

results_cheat = pd.DataFrame(data=cheat_results,
                             columns=['bACC', "Fold", "Effect",
                                      "Type", "Example", "Name"])
results_cheat["Method"] = "Cheat"

results_leakage = pd.DataFrame(data=leakage_results,
                               columns=['bACC', "Fold", "Effect",
                                        "Type", "Example", "Name"])
results_leakage["Method"] = "Leakage"

results_notarget = pd.DataFrame(data=notarget_results,
                                columns=['bACC', "Fold", "Effect",
                                         "Type", "Example", "Name"])
results_notarget["Method"] = "No Target"

results = pd.concat([results_none, results_Juharmonize,
                     results_cheat, results_leakage,
                     results_notarget])

# results.to_csv(save_dir+"results_"+clf_name+"_MAREoS_complete.csv")

# %% Plotting resuts

# results = pd.read_csv("/home/nnieto/Nico/Harmonization/results_classification/MAREoS/results_RF_MAREoS_complete.csv")         # noqa
fig, ax = plt.subplots(1, 1, figsize=[20, 15])

harm_methods = ["JuHarmonize", "Cheat", "Leakage", "No Target", "None"]
pal = sbn.cubehelix_palette(len(harm_methods), rot=-.5, light=0.5, dark=0.2)

sbn.swarmplot(
    data=results,
    x="Name", y="bACC", hue="Method",
    hue_order=harm_methods,
    dodge=True, ax=ax,
    palette=pal
)

sbn.boxplot(
    data=results, color="w", zorder=1,
    x="Name", y="bACC", hue="Method",
    hue_order=harm_methods,
    dodge=True, ax=ax, palette=["w"]*len(harm_methods)
)
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles[:len(harm_methods)], labels[:len(harm_methods)])
ax.axhline(0.5, lw=2, color="k",
           ls="--", alpha=0.7, label="Chance level")
plt.grid(axis="y")
plt.show()
# %%

results = pd.read_csv("/home/nnieto/Nico/Harmonization/results_classification/MAREoS/results_SVM_MAREoS_complete.csv")   # noqa
for exp in results["Name"].unique():
    results_harmonize = results[results["Method"] == "JuHarmonize"]
    results_noharmonize = results[results["Method"] == "No Harmonize"]

    print(exp)
    BAC_Without_Mareos = results_noharmonize[results_noharmonize["Name"] == exp]["bACC"].mean() * 100   # noqa
    BAC = results_harmonize[results_harmonize["Name"] == exp]["bACC"].mean() * 100                      # noqa

    print(results_harmonize[results_harmonize["Name"] == exp]["bACC"].mean())
    print(results_noharmonize[results_noharmonize["Name"] == exp]["bACC"].mean())                       # noqa
    print("RAC")
    print((BAC_Without_Mareos-BAC) / (BAC_Without_Mareos - 50) * 100)

# %%


def calculate_percentage(labels, sites):
    unique_sites = np.unique(sites)

    for site in unique_sites:
        site_mask = (sites == site)
        total_labels = np.sum(site_mask)
        positive_labels = np.sum(labels[site_mask])

        percentage_of_total_labels = (positive_labels / len(labels)) * 100
        percentage_of_total_site = (positive_labels / total_labels) * 100 if total_labels > 0 else 0

        print(f"Site {site}:")
        print(f"  Total Labels: {total_labels}")
        print(f"  Positive Labels: {positive_labels}")
        print(f"  Percentage of Total Labels: {percentage_of_total_labels:.2f}%")
        print(f"  Percentage of Total Site: {percentage_of_total_site:.2f}%")
        print()

# Example usage:
# y = np.array([1, 0, 1, 1, 0, 1, 0, 0, 1, 1])
# site = np.array(['A', 'B', 'A', 'A', 'B', 'C', 'A', 'C', 'B', 'C'])

calculate_percentage(y, site_train)

# %%
import numpy as np

def calculate_percentage(labels, sites):
    unique_sites = np.unique(sites)

    for site in unique_sites:
        site_mask = (sites == site)
        total_labels = np.sum(site_mask)
        positive_labels = np.sum(labels[site_mask])

        percentage_of_total_labels = (positive_labels / len(labels)) * 100
        percentage_of_total_site = (positive_labels / total_labels) * 100 if total_labels > 0 else 0

        print(f"Site {site}:")
        print(f"  Total Labels: {total_labels}")
        print(f"  Positive Labels: {positive_labels}")
        print(f"  Percentage of Total Labels: {percentage_of_total_labels:.2f}%")
        print(f"  Percentage of Total Site: {percentage_of_total_site:.2f}%")
        print()

def filter_data_by_sites(labels, sites, target_sites):
    site_mask = np.isin(sites, target_sites)
    filtered_labels = labels[site_mask]
    filtered_sites = sites[site_mask]

    return filtered_labels, filtered_sites

# Example usage:
y = np.array([1, 0, 1, 1, 0, 1, 0, 0, 1, 1])
site = np.array(['site1', 'site2', 'site1', 'site1', 'site2', 'site3', 'site1', 'site3', 'site2', 'site3'])

calculate_percentage(y, site)

# Example of filtering data for 'site1' and 'site3':
filtered_y, filtered_site = filter_data_by_sites(y, site, target_sites)
print("\nFiltered Data for 'site1' and 'site3':")
print("Labels:", filtered_y)
print("Sites:", filtered_site)



# %%
