# %%
import pandas as pd
import numpy as np
from sklearn import svm         # noqa
import seaborn as sbn
from sklearn.linear_model import ElasticNet  # noqa
from sklearn.model_selection import cross_val_predict, GridSearchCV, KFold  # noqa
import matplotlib.pyplot as plt
from neuroHarmonize import harmonizationLearn, harmonizationApply
from skrvm import RVR
from pathlib import Path
import sys
import seaborn as sns

dir_path = '/home/nnieto/Nico/Harmonization/harmonize_project/3_check_results/'
__file__ = dir_path+'plot_NM.py'
to_append = Path(__file__).resolve().parent.parent.as_posix()
sys.path.append(to_append)

from juharmonize.utils import subset_data # noqa
from lib.harmonize import eval_harmonizer, train_harmonizer  # noqa
# set numpy seed for reproducibility
np.random.seed(42)
# %%
# read data
data_dir = "/home/nnieto/Nico/Harmonization/data/final_data_split/"

features = 3747


x2 = pd.read_csv(data_dir+'X_eNKI.csv')
x2 = x2.iloc[:, 0:features]
y2 = pd.read_csv(data_dir+'Y_eNKI.csv')['age']
site2 = pd.read_csv(data_dir+'Y_eNKI.csv')['site']


x1 = x2.loc[0:409, :]
y1 = y2.loc[0:409]
site1 = site2.loc[0:409]

x2 = x2.loc[410::, :]
y2 = y2.loc[410::]
site2 = site2.loc[410::]

# %%


y = pd.concat([y1, y2])
y = y.to_numpy()

sites_names = pd.concat([site1, site2])

# induce site effect
vars = x1.var(axis=0)
means = x1.mean(axis=0)
# gamma is feature-wise additive effect
gamma = np.random.normal(1, np.sqrt(vars)/5, size=x1.shape[1])
gamma = 0
gamma = np.tile(gamma, (x1.shape[0], 1))
# delta is feature-wise multiplicative effect
delta = np.random.normal(1.2, 0.01, size=x1.shape[1])
delta = 1
delta = np.tile(delta, (x1.shape[0], 1))
# alpha is site-wise additive effect
alpha = 0
# uncombat the data
plt.plot(delta)
x1site = delta*(x1 - alpha) + alpha + gamma

# induce site effect
vars = x2.var(axis=0)
means = x2.mean(axis=0)
# gamma is feature-wise additive effect
gamma = np.random.normal(1, np.sqrt(vars)/5, size=x2.shape[1])
gamma = 0
gamma = np.tile(gamma, (x2.shape[0], 1))
# delta is feature-wise multiplicative effect
delta = np.random.normal(2, np.sqrt(vars)/2, size=x2.shape[1])
delta = 1
delta = np.tile(delta, (x2.shape[0], 1))
# alpha is site-wise additive effect
alpha = 0
# uncombat the data
x2site = delta*(x2 - alpha) + alpha + gamma

sites = np.concatenate((np.zeros(x1.shape[0]), np.ones(x2.shape[0])))

X_unharmonize = pd.concat([x1site, x2site]).to_numpy()
X = pd.concat([x1, x2])

plt.figure(figsize=[15, 7])
plt.scatter(x1site.mean(axis=1), y1)
plt.scatter(x2site.mean(axis=1), y2)

plt.legend(["CamCAN", "eNKI"])        # noqa
plt.xlabel("Average of the features by subject")
plt.ylabel("Subject age")

# %% data plotting
plt.figure(figsize=[15, 7])
plt.plot(x1.mean())
plt.plot(x1site.mean())
plt.title("Feature values for site 1")
plt.legend(["mean of the original feature", "mean of the unharmonized feature"])        # noqa
plt.xlabel("Features")
plt.ylabel("Feature Value")

# plt.figure(figsize=[15, 7])
# plt.plot(x2.mean())
# plt.plot(x2site.mean())
# plt.title("Feature values for site 2")
# plt.legend(["mean of the original feature", "mean of the unharmonized feature"])        # noqa
# plt.xlabel("Features")
# plt.ylabel("Feature Value")

# # %%

# plt.figure(figsize=[15, 7])
# plt.scatter(x2.mean(axis=1), y2)
# plt.scatter(x2site.mean(axis=1),y2)

# plt.legend(["mean of the original feature", "mean of the unharmonized feature"])        # noqa
# plt.xlabel("Features")
# plt.ylabel("Feature Value")
# # %%
# plt.figure(figsize=[15, 7])
# plt.scatter(x1.mean(axis=1), y1)
# plt.scatter(x1site.mean(axis=1),y1)

# plt.legend(["mean of the original feature", "mean of the unharmonized feature"])        # noqa
# plt.xlabel("Features")
# plt.ylabel("Feature Value")
#  %%
plt.figure()
sbn.swarmplot(x=y1)
sbn.swarmplot(x=y2)

# %%
plt.figure(figsize=[15, 7])
plt.scatter(x1.mean(axis=1), y1)
plt.scatter(x2.mean(axis=1), y2)

plt.legend(["mean of the original feature", "mean of the unharmonized feature"])        # noqa
plt.xlabel("Mean Features by subject")
plt.ylabel("Subject age")
# %%
kf = KFold(n_splits=5)
X = pd.concat([x1, x2])

C = np.mean(np.sqrt(np.sum(X**2, axis=1)))
model = svm.SVR(kernel="linear")
print("Normal fit")
pred = cross_val_predict(model, X, y, cv=kf)

results_original = pd.DataFrame({'y_pred': (pred)})
results_original["y_true"] = y
results_original["site"] = sites
results_original["Harmonization Mode"] = "Original"

print("Noise fit")
C = np.mean(np.sqrt(np.sum(X_unharmonize**2, axis=1)))
model = svm.SVR(kernel="linear")
predx = cross_val_predict(model, X_unharmonize, y, cv=kf)

results_unharmonize = pd.DataFrame({'y_pred': (predx)})
results_unharmonize["y_true"] = y
results_unharmonize["site"] = sites
results_unharmonize["Harmonization Mode"] = "Noise Data"
#
plt.figure()
sbn.scatterplot(data=results_original, x="y_true", y="y_pred", hue="site")
mae = np.mean(np.abs(results_original["y_pred"] - results_original["y_true"]))
plt.title("Original data MAE:" + str(mae))

plt.figure()
sbn.scatterplot(data=results_unharmonize, x="y_true", y="y_pred", hue="site")
mae = np.mean(np.abs(results_unharmonize["y_pred"] - results_unharmonize["y_true"]))            # noqa
plt.title("UnHarmonize MAE:" + str(mae))


# %% CHEAT
# harmonization
sites = np.concatenate((np.zeros(x1.shape[0]), np.ones(x2.shape[0])))
covars = pd.DataFrame(sites.astype(int), columns=['SITE'])
covars['AGE'] = y
harm_model, harm_data = harmonizationLearn(X_unharmonize, covars)

model = svm.SVR(kernel="linear")
predh = cross_val_predict(model, harm_data, y, cv=kf)

results_Cheat = pd.DataFrame({'y_pred': (predh)})
results_Cheat["y_true"] = y
results_Cheat["site"] = sites
results_Cheat["Harmonization Mode"] = "Cheat"

plt.figure()
sbn.scatterplot(data=results_Cheat, x="y_true", y="y_pred", hue="site")
mae = np.mean(np.abs(results_Cheat["y_pred"] - results_Cheat["y_true"]))
plt.title("Cheat MAE:" + str(mae))

# %% JuHarmonize
from sklearn.ensemble import RandomForestRegressor          # noqa

pred_model = RVR(kernel="poly", degree=1)
stack_model = RVR(kernel="poly", degree=1)
pred_model = svm.SVR(kernel="linear")
stack_model = svm.SVR(kernel="linear")

out_fold_list = []
y_test_list = []
sites_test_list = []
covars = None
regression_params = {
    "regression_points": 10,
    "regression_search_tol": 2.0,
    "regression_search": True,
}
kf = KFold(n_splits=5)
sites = np.concatenate((np.zeros(x1.shape[0]), np.ones(x2.shape[0])))
for i_fold, (train_index, test_index) in enumerate(kf.split(X_unharmonize)):
    print(i_fold)
    X_train, sites_train, y_train, covars_train, _ = subset_data(train_index, X_unharmonize, sites, y, covars)      # noqa
    X_test, sites_test, y_test, covars_test, _ = subset_data(test_index, X_unharmonize, sites, y, covars)           # noqa

    harm_model = train_harmonizer(
        "pretend",
        X_train,
        y_train,
        sites_train,
        covars_train,
        pred_model=pred_model,

        stack_model=stack_model,
        random_state=42,
        regression_params=regression_params,
        n_splits=5
    )

    out_fold, acc_fold = eval_harmonizer(
        harm_model, X_test, y_test, sites_test, covars_test
    )
    out_fold_list.append(out_fold)
    y_test_list.append(y_test)
    sites_test_list.append(sites_test)
# %% plotting JuHarmonize

results_JuHarmonize = pd.DataFrame({'y_pred': np.concatenate(out_fold_list)})
results_JuHarmonize["y_true"] = np.concatenate(y_test_list)
results_JuHarmonize["site"] = np.concatenate(sites_test_list)
results_JuHarmonize["Harmonization Mode"] = "JuHarmonize"


plt.figure()
sbn.scatterplot(data=results_JuHarmonize, x="y_true", y="y_pred", hue="site")
mae = np.mean(np.abs(results_JuHarmonize["y_pred"] - results_JuHarmonize["y_true"]))        # noqa
plt.title("JuHarmonize MAE:" + str(mae))

##############################################################################
# %% Lekeage
sites = np.concatenate((np.zeros(x1.shape[0]), np.ones(x2.shape[0])))
stack_model = svm.SVR(kernel="linear")
pred_leakage_list = []
y_test_list = []
sites_test_list = []
harm_test_data = []
for i_fold, (train_index, test_index) in enumerate(kf.split(X)):
    print(i_fold)
    covars = None
    X_train, sites_train, y_train, covars_train, _ = subset_data(train_index, X_unharmonize, sites, y, covars)      # noqa
    X_test, sites_test, y_test, covars_test, _ = subset_data(test_index, X_unharmonize, sites, y, covars)           # noqa
    covars_train = pd.DataFrame(sites_train.astype(int), columns=['SITE'])
    covars_train['AGE'] = y_train
    # Fit and transorm train data
    harm_model, harm_data = harmonizationLearn(X_train, covars_train)
    # Fit the model with the harmonizezd trian
    model.fit(harm_data, y_train)

    # Testing
    covars_test = pd.DataFrame(sites_test.astype(int), columns=['SITE'])
    covars_test['AGE'] = y_test
    # Transform the test data
    harm_data = harmonizationApply(X_test, covars_test, harm_model)
    # generate a prediction
    harm_test_data.append(harm_data)
    pred_leakage = model.predict(harm_data)
    pred_leakage_list.append(pred_leakage)
    y_test_list.append(y_test)
    sites_test_list.append(sites_test)


results_leakage = pd.DataFrame({'y_pred': np.concatenate(pred_leakage_list)})
results_leakage["y_true"] = np.concatenate(y_test_list)
results_leakage["site"] = np.concatenate(sites_test_list)
results_leakage["Harmonization Mode"] = "Leakage"
plt.figure()
sbn.scatterplot(data=results_leakage, x="y_true", y="y_pred", hue="site")
mae = np.mean(np.abs(results_leakage["y_pred"] - results_leakage["y_true"]))
plt.title("Leakage MAE:" + str(mae))
# %%

print("MAE Original: " + str(round(np.mean(np.abs(results_original["y_pred"] - results_original["y_true"])),3)))             # noqa
print("MAE UnHarmonize: " + str(round(np.mean(np.abs(results_unharmonize["y_pred"] - results_unharmonize["y_true"])),3)))    # noqa

print("MAE Cheat: " + str(round(np.mean(np.abs(results_Cheat["y_pred"] - results_Cheat["y_true"])),3)))                      # noqa

print("MAE Leak: " + str(round(np.mean(np.abs(results_leakage["y_pred"] - results_leakage["y_true"])),3)))                   # noqa
print("MAE JuHarmonize: " + str(round(np.mean(np.abs(results_JuHarmonize["y_pred"] - results_JuHarmonize["y_true"])),3)))    # noqa


# %%

def data_naming(data):
    data.rename(columns={"site": "Site"}, inplace=True)
    data["Site"].replace({0: "CamCAN", 1: "eNKI"}, inplace=True)
    return data


# Create a figure with 2 rows and 2 columns
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15, 12))

# Create a list of titles for the subplots
titles = ['None', 'Cheat',
          'JuHarmonize', 'Leakage']

data_list = [results_unharmonize, results_Cheat,
             results_JuHarmonize, results_leakage]


# Iterate over the data arrays and plot the scatterplots
for i, ax in enumerate(axes.flat):

    data_harm = data_list[i]
    data = data_naming(data_harm)

    sns.scatterplot(data=data, x="y_true", y="y_pred", hue="Site", ax=ax)
    mae = np.mean(np.abs(data["y_pred"] - data["y_true"]))
    ax.set_title(titles[i] + " MAE: " + r"$\bf{" + str(round(mae, 4)) + "}$")
    ax.set_xlabel("True Age")
    ax.set_ylabel("Predicted Age")
    ax.axline([17, 17],
              [90, 90],
              ls='--', color='black')
    ax.set_xlim(17, 90)
    ax.set_ylim(5, 128)

# Adjust the spacing between subplots
plt.tight_layout()
# Show the plot
plt.show()

# %%
