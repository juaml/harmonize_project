# %% Library import
import pandas as pd
from sklearn.model_selection import KFold
from juharmonize.utils import subset_data
from juharmonize import JuHarmonize
import numpy as np
import seaborn as sbn
from sklearn.svm import SVR, LinearSVR
from sklearn.linear_model import LinearRegression                   # noqa
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
from pathlib import Path
import sys
dir_path = '/home/nnieto/Nico/Harmonization/harmonize_project/lib/'
__file__ = dir_path+'plot_NM.py'
to_append = Path(__file__).resolve().parent.parent.as_posix()
sys.path.append(to_append)
from lib.harmonize import train_harmonizer, eval_harmonizer         # noqa
from lib.io import keep_n_images_by_site
# %% Data loading
# Replace to your data location

data_dir = "/home/nnieto/Nico/Harmonization/data/final_data_split/"
X = []
y = []

# site to load / random noise mean / std noise
datasets = [["eNKI", None, None],
            ["CamCAN", None, None],
            ["SALD", None, None],
            ["DLBS", None, None],
            ["ID1000", None, None],
            ["PIOP1", None, None],
            ["PIOP2", None, None],
            ["1000Gehirne", None, None],
            ["IXI", None, None],
            ["OASIS3", None, None]]

# datasets = [["eNKI", 50, .5],
#             ["CamCAN", None, None]]
# add noise
for dataset, random_mean, noise_scale in datasets:
    print("Loading " + dataset)
    data = pd.read_csv(data_dir + "X_"+dataset+".csv")
    y.append(pd.read_csv(data_dir + "Y_"+dataset+".csv"))

    if random_mean is None:
        noisy_df = data
        print(dataset+" No noise")
    else:
        # save the same targets
        noisy_df = pd.DataFrame()
        # Iterate over each column (feature) in the original dataframe
        for feature in data.columns:
            # Generate random mean (-1 or 1)
            if random_mean < 0:
                random_mean = np.random.choice([-random_mean, random_mean])

            if noise_scale is not None:
                # Generate random noise using the Gaussian distribution
                # with mean and standard deviation of 1
                noise = np.random.normal(loc=random_mean,
                                         scale=noise_scale,
                                         size=len(data))

            else:
                noise = random_mean

            # Add the noise to the original feature values
            # and store in the new dataframe
            noisy_df[feature] = data[feature] + noise
        print(dataset+" with noise")

    X.append(noisy_df)

# Concatenate the data
X = pd.concat(X, axis=0)
# Drop the last NAN column
X.dropna(axis=1, inplace=True)
# Put to numpy for sklearn
X = X.to_numpy()
# Concatenate the y
y = pd.concat(y, axis=0)
# Get sites
sites = y["site"].to_numpy()
# Get age as target
y = y["age"].to_numpy()
# %%
colvar = np.var(X, axis=0)
# Identify variables with low variance
var_ix = colvar > 10e-5
idxvar = np.argsort(-colvar)
# Delet variables with low variance
idxvar = idxvar[var_ix]
X = X[:, idxvar]

# No covars
covars = None
# %%
# images_by_site = 20
# X, y, sites = keep_n_images_by_site(images_by_site, X, y, sites)
# %% Model set up
# Kfold
kf = KFold(n_splits=5, shuffle=True, random_state=23)
# models definition
# pred_model = SVR(kernel="linear", shrinking=False)
# pred_model_cheat = SVR(kernel="linear", C=1)
# pred_model = SVR(kernel="linear",  C=1)

# pred_model_cheat = LinearSVR(max_iter=2000)
# pred_model = LinearSVR(max_iter=2000)

pred_model_cheat = LinearRegression()
pred_model = LinearRegression()
# Random state
random_state = 23

# Initialize variables
y_true = []
predictions_none = []
predictions_cheat = []
predictions_none_juharmonize = []
sites_all = []

# Transform X in cheat scheme
cheat_model = JuHarmonize()
X_cheat = cheat_model.fit_transform(X, y, sites, covars)
# %%
for i_fold, (train_index, test_index) in enumerate(kf.split(X)):
    print(i_fold+1)
    # NONE METHOD
    # get train from pooled data
    X_train, sites_train, y_train, covars_train, _ = subset_data(
        train_index, X, sites, y, covars
    )
    # None Prediction
    pred_model.fit(X_train, y_train)
    # get test from pooled data
    X_test, sites_test, y_test, covars_test, _ = subset_data(
        test_index, X, sites, y, covars
    )
    predictions_none.append(pred_model.predict(X_test))

    # CHEAT METHOD
    # get train from cheat data
    X_train_cheat, sites_train, y_train, covars_train, _ = subset_data(
        train_index, X_cheat, sites, y, covars
    )
    # Cheat Prediction
    pred_model_cheat.fit(X_train_cheat, y_train)

    # get test from cheat data
    X_test_cheat, sites_test, y_test, covars_test, _ = subset_data(
        test_index, X_cheat, sites, y, covars
    )
    predictions_cheat.append(pred_model_cheat.predict(X_test_cheat))

    # GENERAL INFORMATION
    y_true.append(y_test)
    sites_all.append(sites_test)

# %% Generate dataframes
data_none = pd.DataFrame({'y_pred': np.concatenate(predictions_none)})
data_none["y_true"] = np.concatenate(y_true)
data_none["site"] = np.concatenate(sites_all)
data_none["Harmonization Mode"] = "None"

sbn.scatterplot(data=data_none, x="y_true", y="y_pred", hue="site")
plt.title("None MAE: " + str(mean_absolute_error(data_none["y_true"], data_none["y_pred"]))) # noqa
# %%
data_cheat = pd.DataFrame({'y_pred': np.concatenate(predictions_cheat)})
data_cheat["y_true"] = np.concatenate(y_true)
data_cheat["site"] = np.concatenate(sites_all)
data_cheat["Harmonization Mode"] = "Cheat"
sbn.scatterplot(data=data_cheat, x="y_true", y="y_pred", hue="site")
plt.title("Cheat MAE: " + str(mean_absolute_error(data_cheat["y_true"], data_cheat["y_pred"]))) # noqa)


data = pd.concat([data_cheat, data_none])
# %% For GraphNone
absolute = False

if absolute:
    data["y_diff"] = np.abs(data["y_true"]-data["y_pred"])
else:
    data["y_diff"] = data["y_true"]-data["y_pred"]

fig, ax = plt.subplots(1, 1, figsize=[20, 10])
pal = sbn.cubehelix_palette(5, rot=-.15, light=0.85, dark=0.3)

sbn.boxenplot(
    data=data, palette=pal,
    x="site", y="y_diff", hue="Harmonization Mode")
plt.ylabel("Age prediction difference [years]")
plt.title("Age Prediction")
plt.xlabel("Sites ID")
plt.grid(alpha=0.5, axis="y", c="black")
plt.show()

# %% Scatter predictions for one method
method = "None"
data_toplot = data[data["Harmonization Mode"] == method]
sbn.scatterplot(data=data_none, x="y_true", y="y_pred", hue="site")
plt.title(method)
# %%
method = "Cheat"
data_toplot = data[data["Harmonization Mode"] == method]
sbn.scatterplot(data=data_toplot, x="y_true", y="y_pred", hue="site")
plt.title(method)
# %%
coef_100 = pred_model.coef_
# %%
coef_1 = pred_model.coef_

# %%
