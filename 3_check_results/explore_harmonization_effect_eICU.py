# %%
import seaborn as sbn
import pandas as pd
import numpy as np
from juharmonize import JuHarmonize
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler

SMALL_SIZE = 15
MEDIUM_SIZE = 20
BIGGER_SIZE = 30

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

median_sort = True
scaler_bool = False
median_line_alpha = 0.7
min_images_per_site = 30

# Data loading
root_dir = "/home/nnieto/Nico/Harmonization/data/eICU/"
data = pd.read_csv(root_dir + "equals_to_paper_data.csv", index_col=0)
ABG_of_interes = ["paO2", "paCO2", "pH", "Base Excess",
                  "Hgb", "glucose", "bicarbonate", "lactate"]

features_to_plot = ["Base Excess"]

# Processing
X = data.loc[:, ABG_of_interes].to_numpy()
sites = data["site"].to_numpy()
Y = data["endpoint"].replace({"Expired": 1, "Alive": 0}).to_numpy()
cheat_model = JuHarmonize(preserve_target=False)
scaler = RobustScaler()

# Remove sites with less than a thd number of patients
site_counts = data["site"].value_counts()

# filter the site_ids with less than a thd
mask = site_counts[site_counts > min_images_per_site].index.tolist()

# Filter the sites with the minimun number of patietes
data_filter = data[data['site'].isin(mask)]

X = data_filter.loc[:, ABG_of_interes].to_numpy()
sites = data_filter["site"].to_numpy()
Y = data_filter["endpoint"].replace({"Expired": 1, "Alive": 0}).to_numpy()

if scaler_bool:
    X = scaler.fit_transform(X, Y)
X_cheat = cheat_model.fit_transform(X, Y, sites)
X_cheat_df = pd.DataFrame(X_cheat, columns=ABG_of_interes)
X_cheat_df["site"] = sites

data_filter.loc[:, ABG_of_interes] = X

X_diff_df = pd.DataFrame(X_cheat - X, columns=ABG_of_interes)
X_diff_df["site"] = sites

for feature in features_to_plot:
    if median_sort:
        # Calculate global median
        global_median = data_filter[feature].median()

        # Calculate median for each site and the difference from global median
        medians = data_filter.groupby('site')[feature].median()
        medians_diff = np.abs(medians - global_median)

        # Sort sites by median difference
        sorted_sites = medians_diff.sort_values(ascending=False).index
    else:
        sorted_sites = np.unique(data_filter["site"])
    fig, ax = plt.subplots(3, 1, figsize=[20, 20])
    ax[0] = sbn.boxplot(data=data_filter, y=feature, x="site", ax=ax[0],
                        order=sorted_sites)
    ax[0].axhline(data_filter.loc[:, feature].median(), lw=2, color="k",
                  ls="--", alpha=median_line_alpha, label="Global Median")
    y_lim = ax[0].get_ylim()
    ax[0].set_title("Raw feature")
    ax[0].legend()

    ax[1] = sbn.boxplot(data=X_cheat_df, y=feature, x="site", ax=ax[1],
                        order=sorted_sites)
    ax[1].axhline(X_cheat_df.loc[:, feature].median(), lw=2, color="k",
                  ls="--", alpha=median_line_alpha, label="Global Median")
    ax[1].legend()
    ax[1].set_ylim(y_lim)
    ax[1].set_title('Harmonize feature')

    sbn.boxplot(data=X_diff_df, y=feature, x="site", ax=ax[2],
                order=sorted_sites)
    ax[2].axhline(0, lw=2, color="k",
                  ls="--", alpha=median_line_alpha, label="0 difference")
    ax[2].legend()
    ax[2].set_title('Harmonization Effect (Harmonize Feature - Raw Feature)')


# %%

# %%
