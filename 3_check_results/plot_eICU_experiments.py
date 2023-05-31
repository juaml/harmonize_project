# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.metrics import (
    balanced_accuracy_score,
    roc_auc_score,
)


def get_auc_for_plot(y_pred, y_test_list, sites_test_list, folds_lits):

    test_plot = np.concatenate(y_test_list)
    y_pred_0 = np.concatenate([p[:, 0] for p in y_pred])
    y_pred_1 = np.concatenate([p[:, 1] for p in y_pred])
    site_plot = np.concatenate(sites_test_list)
    fold_plot = np.concatenate(folds_lits)
    # create DataFrame with the true labels,
    # predicted probabilities and site list
    df_full = pd.DataFrame({'y_test': test_plot,
                            'y_pred_0': y_pred_0,
                            'y_pred_1': y_pred_1,
                            'site': site_plot,
                            'fold': fold_plot})

    #
    # create a list of the unique sites in the data
    # create a list of the unique sites in the data
    sites = df_full['site'].unique()
    folds = np.unique(df_full["fold"])
    # create a dictionary to store the AUC for each site
    auc_list = []

    # calculate AUC for each site and store it in the dictionary
    for fold in folds:
        df = df_full[df_full["fold"] == fold]
        for site in sites:
            if df[df['site'] == site]['y_test'].nunique() == 2:
                auc = roc_auc_score(df[df['site'] == site]['y_test'],
                                    df[df['site'] == site]['y_pred_1'])

                acc = balanced_accuracy_score(df[df['site'] == site]['y_test'],
                                              np.round(df[df['site'] == site]
                                                       ['y_pred_1']))
                auc_list.append([site, fold, auc, acc])

            else:
                auc_list.append([site, fold, 0.5, 0.5])

    # create a DataFrame with the AUC values for each site
    return pd.DataFrame(auc_list, columns=["site", "fold",
                                           "AUC", "Balanced_ACC"])


# create example data
data_dir = "/home/nnieto/Nico/Harmonization/data/eICU/Results/10_images_min/"

# Create the list of lists and save names
save_list = [[data_dir+"eICU_pred_none", ""],
             [data_dir+"eICU_pred_target", ""],
             [data_dir+"eICU_pred_notarget", ""],
             [data_dir+"eICU_pred_pretend", ""],
             [data_dir+"eICU_pred_cheat", ""],
             [data_dir+"eICU_y_test_list", ""],
             [data_dir+"eICU_sites_test_list", ""],
             [data_dir+"eICU_fold_plot", ""],
             [data_dir+"eICU_pred_none", ""],
             [data_dir+"eICU_pred_target", ""]]

# Load the data objects from the files
loaded_data = {}
for save_name, _ in save_list:
    with open(save_name + '.pickle', 'rb') as file:
        loaded_data[save_name] = pickle.load(file)

# Change the variable name as previous
for save_name, data in loaded_data.items():
    name = save_name.split(sep="/eICU_")[1]
    exec(f"{name} = data")

data = pd.read_csv(data_dir + "eICU_data_used.csv", index_col=0)
X_cheat = pd.read_csv(data_dir + "eICU_data_harmonized_cheat_used.csv",
                      index_col=0).to_numpy()

# %%
ABG_of_interes = ["paO2", "paCO2", "pH", "Base Excess",
                  "Hgb", "glucose", "bicarbonate", "lactate"]
X = data.loc[:, ABG_of_interes].to_numpy()

auc_none = get_auc_for_plot(pred_none, y_test_list,             # noqa
                            sites_test_list, fold_plot)         # noqa

auc_harm = get_auc_for_plot(pred_cheat, y_test_list,            # noqa
                            sites_test_list, fold_plot)         # noqa
# %%
fig, ax = plt.subplots(3, 1, figsize=[30, 20])

metric_plot = "AUC"
sort_by = "mean_performance"
# Calculate global median

if sort_by == "images":
    sorted_sites = data["site"].value_counts().index

elif sort_by == "mean_performance":
    # Calculate median for each site and the difference from global median
    global_median = auc_none[metric_plot].mean()
    medians = auc_none.groupby('site')[metric_plot].mean()
    medians_diff = medians - global_median
    # Sort sites by median difference
    sorted_sites = medians_diff.sort_values(ascending=False).index

elif sort_by == "expired_count":
    sorted_sites = \
        data[data["endpoint"] == "Expired"]["site"].value_counts().index

elif sort_by == "feature_changes":
    x_diff = np.abs(X - X_cheat)
    x_diff = pd.DataFrame(x_diff, columns=ABG_of_interes)

    x_diff["site"] = data["site"]
    global_median = x_diff.mean()

    medians = x_diff.groupby('site').mean()
    medians_diff = medians - global_median
    medians_diff["mean"] = medians_diff.loc[:, ABG_of_interes].mean()
    sorted_sites = medians_diff.sort_values(by="mean", ascending=False).index


# create a barplot of the AUC values for each site
sns.barplot(data=auc_none, x='site', y=metric_plot,
            ax=ax[0], order=sorted_sites)
ax[0].axhline(0.5, lw=2, color="k",
              ls="--", alpha=0.7, label="Chance level")
ax[0].axhline(auc_none[metric_plot].mean(), lw=2, color="k",
              ls="-", alpha=0.5, label="Mean "+metric_plot)
ax[0].set_title("Performance with Raw feature")

ax[0].legend()

sns.barplot(data=auc_harm, x='site', y=metric_plot,
            ax=ax[1], order=sorted_sites)
ax[1].axhline(0.5, lw=2, color="k",
              ls="--", alpha=0.7, label="Chance level")
ax[1].axhline(auc_harm[metric_plot].mean(), lw=2, color="k",
              ls="--", alpha=0.5, label="Harmonized Mean "+metric_plot)
ax[1].axhline(auc_none[metric_plot].mean(), lw=2, color="k",
              ls="-", alpha=0.5, label="None harmonization Mean "+metric_plot)
ax[1].set_title("Performance with Harmonized feature (JuHarmonize model)")

ax[1].legend()

auc_df_diff = auc_harm.copy()
auc_df_diff[metric_plot] = auc_harm[metric_plot] - auc_none[metric_plot]
sns.barplot(data=auc_df_diff, x='site', y=metric_plot,
            ax=ax[2], order=sorted_sites)
ax[2].axhline(auc_df_diff[metric_plot].mean(), lw=2, color="k",
              ls="--", alpha=0.5, label="Mean difference")
ax[2].axhline(0, lw=2, color="k",
              ls="-", alpha=0.7, label="0 level")
ax[2].set_title("Performance difference (Harmonized - None)")

ax[2].legend()

plt.show()

# %%
