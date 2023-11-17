# %%
import warnings
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

root_dir = "/home/nnieto/Nico/Harmonization/data/MAREoS/public_datasets/"
save_dir = "/home/nnieto/Nico/Harmonization/results_classification/MAREoS/"

effects = ["true", "eos"]
effect_types = ["simple", "interaction"]
effect_examples = ["1", "2"]

clf_name = "SVM"
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
harm_results = []
simple_results = []
cheat_results = []
leakage_results = []
notarget_results = []
for effect in effects:
    for e_types in effect_types:
        for e_example in effect_examples:

            example = effect+"_"+e_types+e_example
            print("Experiment name: " + example)
            data = pd.read_csv(root_dir+example+"_data.csv", index_col=0)
            target = pd.read_csv(root_dir+example+"_response.csv", index_col=0)

            folds = data["folds"]
            sites = data["site"]
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
                harm_results.append([balanced_accuracy_score(y_true=y_test,
                                                             y_pred=y_pred), fold, effect, e_types, e_example, example])                     # noqa


# Save results
results_none = pd.DataFrame(data=simple_results, columns=['bACC', "Fold", "Effect", "Type", "Example", "Name"])                      # noqa
results_none["Method"] = "None"
results_Juharmonize = pd.DataFrame(data=harm_results, columns=['bACC', "Fold", "Effect", "Type", "Example", "Name"])                          # noqa
results_Juharmonize["Method"] = "JuHarmonize"
results_cheat = pd.DataFrame(data=cheat_results, columns=['bACC', "Fold", "Effect", "Type", "Example", "Name"])                          # noqa
results_cheat["Method"] = "Cheat"
results_leakage = pd.DataFrame(data=leakage_results, columns=['bACC', "Fold", "Effect", "Type", "Example", "Name"])                          # noqa
results_leakage["Method"] = "Leakage"
results_notarget = pd.DataFrame(data=notarget_results, columns=['bACC', "Fold", "Effect", "Type", "Example", "Name"])                          # noqa
results_notarget["Method"] = "No Target"

results = pd.concat([results_none, results_Juharmonize,
                     results_cheat, results_leakage,
                     results_notarget])
results.to_csv(save_dir+"results_"+clf_name+"_MAREoS_complete.csv")

# %% Plotting resuts

# results = pd.read_csv("/home/nnieto/Nico/Harmonization/results_classification/MAREoS/results_SVM_MAREoS.csv")         # noqa
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

results = pd.read_csv("/home/nnieto/Nico/Harmonization/results_classification/MAREoS/results_SVM_MAREoS.csv")   # noqa
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
